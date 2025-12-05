"""Dataset management service for Kaggle and HuggingFace integration."""

import os
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

import pandas as pd
from huggingface_hub import HfApi, list_datasets, DatasetCard


class DatasetService:
    """Service for browsing and downloading datasets from Kaggle and HuggingFace."""

    def __init__(self, use_postgres: bool = True):
        self.kaggle_token = os.environ.get("KAGGLE_API_TOKEN")
        self.hf_token = os.environ.get("HUGGINGFACE_API_TOKEN")
        self.data_dir = Path(os.environ.get("DATA_DIR", "data"))
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.use_postgres = use_postgres

        # Initialize HuggingFace API
        self.hf_api = HfApi(token=self.hf_token) if self.hf_token else HfApi()

        # Setup Kaggle credentials
        self._setup_kaggle_credentials()

    def _setup_kaggle_credentials(self) -> None:
        """Setup Kaggle credentials from environment variable."""
        if not self.kaggle_token:
            return

        kaggle_dir = Path.home() / ".kaggle"
        kaggle_dir.mkdir(exist_ok=True)

        kaggle_json = kaggle_dir / "kaggle.json"
        if not kaggle_json.exists():
            # Parse token (expected format: {"username":"xxx","key":"xxx"})
            try:
                creds = json.loads(self.kaggle_token)
                kaggle_json.write_text(json.dumps(creds))
                os.chmod(kaggle_json, 0o600)
            except json.JSONDecodeError:
                # Assume it's just the API key
                pass

    # =========================================================================
    # HuggingFace Methods
    # =========================================================================

    def search_hf_datasets(
        self, query: str = "", task: Optional[str] = None, limit: int = 20, sort: str = "downloads"
    ) -> List[Dict[str, Any]]:
        """Search HuggingFace datasets.

        Args:
            query: Search query string
            task: Filter by task (e.g., 'text-classification', 'sentiment-analysis')
            limit: Maximum number of results
            sort: Sort by ('downloads', 'likes', 'created_at')

        Returns:
            List of dataset info dictionaries
        """
        try:
            datasets = list(
                list_datasets(
                    search=query if query else None,
                    task_categories=task,
                    sort=sort,
                    direction=-1,  # Descending
                    limit=limit,
                )
            )

            results = []
            for ds in datasets:
                results.append(
                    {
                        "id": ds.id,
                        "author": ds.author,
                        "name": ds.id.split("/")[-1] if "/" in ds.id else ds.id,
                        "downloads": getattr(ds, "downloads", 0),
                        "likes": getattr(ds, "likes", 0),
                        "tags": getattr(ds, "tags", []),
                        "created_at": str(getattr(ds, "created_at", "")),
                        "source": "huggingface",
                        "url": f"https://huggingface.co/datasets/{ds.id}",
                    }
                )

            return results
        except Exception as e:
            return [{"error": str(e)}]

    def get_hf_dataset_info(self, dataset_id: str) -> Dict[str, Any]:
        """Get detailed information about a HuggingFace dataset.

        Args:
            dataset_id: Dataset identifier (e.g., 'emotion', 'tweet_eval')

        Returns:
            Dataset information dictionary
        """
        try:
            info = self.hf_api.dataset_info(dataset_id)

            # Try to get the dataset card
            card_content = ""
            try:
                card = DatasetCard.load(dataset_id)
                card_content = card.text[:2000] if card.text else ""
            except Exception:
                pass

            return {
                "id": info.id,
                "author": info.author,
                "downloads": getattr(info, "downloads", 0),
                "likes": getattr(info, "likes", 0),
                "tags": getattr(info, "tags", []),
                "description": card_content,
                "created_at": str(getattr(info, "created_at", "")),
                "last_modified": str(getattr(info, "last_modified", "")),
                "source": "huggingface",
                "url": f"https://huggingface.co/datasets/{dataset_id}",
            }
        except Exception as e:
            return {"error": str(e), "id": dataset_id}

    def download_hf_dataset(
        self, dataset_id: str, subset: Optional[str] = None, split: str = "train"
    ) -> Dict[str, Any]:
        """Download a HuggingFace dataset and store in PostgreSQL.

        Args:
            dataset_id: Dataset identifier
            subset: Dataset subset/configuration (e.g., 'sentiment' for tweet_eval)
            split: Dataset split to download

        Returns:
            Download result with stats
        """
        try:
            from datasets import load_dataset

            # Load dataset
            if subset:
                dataset = load_dataset(dataset_id, subset, split=split)
            else:
                dataset = load_dataset(dataset_id, split=split)

            df = pd.DataFrame(dataset)
            
            # Store in PostgreSQL if enabled
            if self.use_postgres:
                return self._store_dataset_in_postgres(
                    df=df,
                    name=f"hf_{dataset_id.replace('/', '_')}_{subset or split}",
                    source="huggingface",
                    source_id=dataset_id,
                    subset=subset,
                    split=split,
                )
            
            # Fallback to parquet (for local development without DB)
            output_dir = self.data_dir / "raw" / "hf" / dataset_id.replace("/", "_")
            output_dir.mkdir(parents=True, exist_ok=True)

            filename = f"{subset}_{split}" if subset else split
            output_path = output_dir / f"{filename}.parquet"

            df.to_parquet(output_path, index=False)

            return {
                "success": True,
                "dataset_id": dataset_id,
                "subset": subset,
                "split": split,
                "path": str(output_path),
                "num_rows": len(df),
                "columns": list(df.columns),
                "size_mb": output_path.stat().st_size / 1024 / 1024,
            }
        except Exception as e:
            return {"success": False, "error": str(e), "dataset_id": dataset_id}
    
    def _store_dataset_in_postgres(
        self,
        df: pd.DataFrame,
        name: str,
        source: str,
        source_id: str,
        subset: Optional[str] = None,
        split: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Store a DataFrame in PostgreSQL.
        
        Args:
            df: DataFrame to store
            name: Unique dataset name
            source: Source (kaggle, huggingface, twitter)
            source_id: Original dataset ID
            subset: Dataset subset
            split: Dataset split
            
        Returns:
            Result dict with success status
        """
        try:
            from app.models.dataset import DatasetRepository
            
            # Detect text and label columns
            text_col = None
            label_col = None
            
            # Common text column names
            text_candidates = ['text', 'sentence', 'content', 'tweet', 'review', 'comment']
            for col in text_candidates:
                if col in df.columns:
                    text_col = col
                    break
            
            # Common label column names
            label_candidates = ['label', 'sentiment', 'emotion', 'target', 'class']
            for col in label_candidates:
                if col in df.columns:
                    label_col = col
                    break
            
            if not text_col:
                # Use first string column
                for col in df.columns:
                    if df[col].dtype == 'object':
                        text_col = col
                        break
            
            if not text_col:
                return {"success": False, "error": "No text column found in dataset"}
            
            # Create dataset metadata
            dataset = DatasetRepository.create_dataset(
                name=name,
                source=source,
                source_id=source_id,
                subset=subset,
                split=split,
                columns=list(df.columns),
                description=f"Downloaded from {source}: {source_id}",
            )
            
            # Clear existing samples (for re-import)
            DatasetRepository.clear_samples(dataset.id)
            
            # Prepare samples
            samples = []
            label_mapping = {}
            
            for idx, row in df.iterrows():
                text = str(row[text_col]) if text_col else ""
                label = row.get(label_col) if label_col else None
                
                # Handle label mapping for text labels
                label_text = None
                if label is not None:
                    if isinstance(label, str):
                        if label not in label_mapping:
                            label_mapping[label] = len(label_mapping)
                        label_text = label
                        label = label_mapping[label]
                    else:
                        label = int(label)
                
                samples.append({
                    "text": text,
                    "label": label,
                    "label_text": label_text,
                    "extra_data": {col: str(row[col]) for col in df.columns 
                                if col not in [text_col, label_col]},
                })
            
            # Add samples in batches
            count = DatasetRepository.add_samples(dataset, samples, batch_size=1000)
            
            return {
                "success": True,
                "dataset_id": source_id,
                "name": name,
                "subset": subset,
                "split": split,
                "storage": "postgresql",
                "num_rows": count,
                "columns": list(df.columns),
                "text_column": text_col,
                "label_column": label_col,
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}

    # =========================================================================
    # Kaggle Methods
    # =========================================================================

    def search_kaggle_datasets(
        self, query: str = "", sort_by: str = "hottest", file_type: str = "csv", limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Search Kaggle datasets.

        Args:
            query: Search query
            sort_by: Sort method ('hottest', 'votes', 'updated', 'active')
            file_type: Filter by file type
            limit: Maximum results

        Returns:
            List of dataset info dictionaries
        """
        try:
            # Note: --max-size expects bytes, not "10GB" format
            # 10GB = 10 * 1024 * 1024 * 1024 = 10737418240 bytes
            result = subprocess.run(
                [
                    "kaggle",
                    "datasets",
                    "list",
                    "-s",
                    query,
                    "--sort-by",
                    sort_by,
                    "--file-type",
                    file_type,
                    "-v",
                    "--csv",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                return [{"error": result.stderr or "Kaggle API error"}]

            lines = result.stdout.strip().split("\n")
            if len(lines) < 2:
                return []

            # Parse CSV output
            import csv

            reader = csv.DictReader(lines)

            results = []
            for row in list(reader)[:limit]:
                results.append(
                    {
                        "id": row.get("ref", ""),
                        "name": row.get("title", row.get("ref", "").split("/")[-1]),
                        "author": (
                            row.get("ref", "").split("/")[0] if "/" in row.get("ref", "") else ""
                        ),
                        "size": row.get("size", ""),
                        "downloads": int(row.get("downloadCount", 0)),
                        "votes": int(row.get("voteCount", 0)),
                        "last_updated": row.get("lastUpdated", ""),
                        "source": "kaggle",
                        "url": f"https://www.kaggle.com/datasets/{row.get('ref', '')}",
                    }
                )

            return results
        except subprocess.TimeoutExpired:
            return [{"error": "Kaggle API timeout"}]
        except Exception as e:
            return [{"error": str(e)}]

    def get_kaggle_dataset_info(self, dataset_id: str) -> Dict[str, Any]:
        """Get detailed information about a Kaggle dataset.

        Args:
            dataset_id: Dataset identifier (e.g., 'kazanova/sentiment140')

        Returns:
            Dataset information dictionary
        """
        try:
            result = subprocess.run(
                ["kaggle", "datasets", "metadata", dataset_id, "-p", "/tmp"],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                return {"error": result.stderr, "id": dataset_id}

            # Read metadata file
            metadata_path = Path("/tmp") / "dataset-metadata.json"
            if metadata_path.exists():
                with open(metadata_path) as f:
                    metadata = json.load(f)
                metadata_path.unlink()
                return {
                    "id": dataset_id,
                    "title": metadata.get("title", ""),
                    "description": metadata.get("description", "")[:2000],
                    "licenses": metadata.get("licenses", []),
                    "keywords": metadata.get("keywords", []),
                    "source": "kaggle",
                    "url": f"https://www.kaggle.com/datasets/{dataset_id}",
                }

            return {"id": dataset_id, "source": "kaggle"}
        except Exception as e:
            return {"error": str(e), "id": dataset_id}

    def download_kaggle_dataset(self, dataset_id: str, unzip: bool = True) -> Dict[str, Any]:
        """Download a Kaggle dataset and store in PostgreSQL.

        Args:
            dataset_id: Dataset identifier (e.g., 'kazanova/sentiment140')
            unzip: Whether to unzip the downloaded files

        Returns:
            Download result with stats
        """
        try:
            # Download to temp directory first
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                cmd = ["kaggle", "datasets", "download", "-d", dataset_id, "-p", str(temp_path)]
                if unzip:
                    cmd.append("--unzip")

                result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

                if result.returncode != 0:
                    return {"success": False, "error": result.stderr, "dataset_id": dataset_id}

                # Find CSV files and load into PostgreSQL
                csv_files = list(temp_path.rglob("*.csv"))
                
                if not csv_files and self.use_postgres:
                    return {"success": False, "error": "No CSV files found in dataset", "dataset_id": dataset_id}
                
                if self.use_postgres and csv_files:
                    # Load the largest CSV (usually the main data file)
                    main_file = max(csv_files, key=lambda f: f.stat().st_size)
                    
                    # Read CSV in chunks for large files
                    chunks = []
                    for chunk in pd.read_csv(main_file, chunksize=100000, 
                                             encoding='latin-1', on_bad_lines='skip'):
                        chunks.append(chunk)
                    
                    df = pd.concat(chunks, ignore_index=True)
                    
                    return self._store_dataset_in_postgres(
                        df=df,
                        name=f"kaggle_{dataset_id.replace('/', '_')}",
                        source="kaggle",
                        source_id=dataset_id,
                    )
                
                # Fallback: copy to data directory (not for large files!)
                output_dir = self.data_dir / "raw" / "kaggle" / dataset_id.replace("/", "_")
                output_dir.mkdir(parents=True, exist_ok=True)
                
                files = []
                total_size = 0
                for f in temp_path.rglob("*"):
                    if f.is_file():
                        size = f.stat().st_size
                        # Skip files larger than 50MB for git
                        if size > 50 * 1024 * 1024:
                            continue
                        total_size += size
                        dest = output_dir / f.name
                        shutil.copy2(f, dest)
                        files.append({"name": f.name, "path": str(dest), "size_mb": size / 1024 / 1024})

                return {
                    "success": True,
                    "dataset_id": dataset_id,
                    "path": str(output_dir),
                    "files": files,
                    "total_size_mb": total_size / 1024 / 1024,
                }
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Download timeout", "dataset_id": dataset_id}
        except Exception as e:
            return {"success": False, "error": str(e), "dataset_id": dataset_id}

    # =========================================================================
    # Local Dataset Management
    # =========================================================================

    def list_local_datasets(self) -> List[Dict[str, Any]]:
        """List all locally stored datasets (PostgreSQL + files).

        Returns:
            List of local dataset information
        """
        datasets = []
        
        # List PostgreSQL datasets first
        if self.use_postgres:
            try:
                from app.models.dataset import DatasetRepository
                
                for ds in DatasetRepository.list_datasets():
                    datasets.append({
                        "name": ds.name,
                        "path": f"postgresql://datasets/{ds.id}",
                        "type": "postgresql",
                        "source": ds.source,
                        "num_files": 1,
                        "num_rows": ds.num_rows,
                        "size_mb": ds.num_rows * 0.001,  # Estimate
                        "columns": ds.columns or [],
                        "created_at": ds.created_at.isoformat() if ds.created_at else None,
                    })
            except Exception:
                pass  # Database not available

        # Also list file-based datasets
        for source_dir in ["raw", "processed"]:
            source_path = self.data_dir / source_dir
            if not source_path.exists():
                continue

            for dataset_dir in source_path.rglob("*"):
                if dataset_dir.is_dir():
                    files = list(dataset_dir.glob("*"))
                    if files:
                        total_size = sum(f.stat().st_size for f in files if f.is_file())
                        datasets.append(
                            {
                                "name": dataset_dir.name,
                                "path": str(dataset_dir),
                                "type": source_dir,
                                "num_files": len([f for f in files if f.is_file()]),
                                "size_mb": total_size / 1024 / 1024,
                                "files": [f.name for f in files if f.is_file()][:10],
                            }
                        )

        return datasets

    def get_dataset_preview(self, path: str, num_rows: int = 10) -> Dict[str, Any]:
        """Get a preview of a dataset file.

        Args:
            path: Path to the dataset file
            num_rows: Number of rows to preview

        Returns:
            Preview data with sample rows and statistics
        """
        path_obj = Path(path)
        if not path_obj.exists():
            return {"error": f"File not found: {path_obj}"}

        try:
            if path_obj.suffix == ".parquet":
                df = pd.read_parquet(path_obj)
            elif path_obj.suffix == ".csv":
                df = pd.read_csv(path_obj, nrows=num_rows * 10)
            elif path_obj.suffix == ".json":
                df = pd.read_json(path_obj, lines=True, nrows=num_rows * 10)
            else:
                return {"error": f"Unsupported file format: {path_obj.suffix}"}

            # Get statistics
            stats = {
                "total_rows": len(df),
                "columns": list(df.columns),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "memory_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
            }

            # Get value counts for categorical columns
            for col in df.columns:
                if df[col].dtype == "object" or df[col].nunique() < 20:
                    stats[f"{col}_distribution"] = df[col].value_counts().head(10).to_dict()

            return {
                "path": str(path_obj),
                "preview": df.head(num_rows).to_dict(orient="records"),
                "stats": stats,
            }
        except Exception as e:
            return {"error": str(e), "path": str(path_obj)}

    def delete_dataset(self, path: str) -> Dict[str, Any]:
        """Delete a local dataset (PostgreSQL or file-based).

        Args:
            path: Path to the dataset directory or postgresql://datasets/{id}

        Returns:
            Deletion result
        """
        # Handle PostgreSQL datasets
        if path.startswith("postgresql://datasets/"):
            if self.use_postgres:
                try:
                    from app.models.dataset import DatasetRepository
                    
                    dataset_id = int(path.split("/")[-1])
                    if DatasetRepository.delete_dataset(dataset_id):
                        return {"success": True, "deleted": path}
                    return {"success": False, "error": "Dataset not found"}
                except Exception as e:
                    return {"success": False, "error": str(e)}
            return {"success": False, "error": "PostgreSQL not enabled"}
        
        # Handle file-based datasets
        path_obj = Path(path)
        if not path_obj.exists():
            return {"success": False, "error": "Path not found"}

        # Safety check - only allow deletion within data directory
        try:
            path_obj.relative_to(self.data_dir)
        except ValueError:
            return {"success": False, "error": "Cannot delete files outside data directory"}

        try:
            if path_obj.is_dir():
                shutil.rmtree(path_obj)
            else:
                path_obj.unlink()
            return {"success": True, "deleted": str(path_obj)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # =========================================================================
    # HuggingFace Upload (for your trained models/datasets)
    # =========================================================================

    def upload_to_huggingface(
        self, local_path: str, repo_id: str, repo_type: str = "dataset", private: bool = False
    ) -> Dict[str, Any]:
        """Upload a local dataset to HuggingFace Hub.

        Args:
            local_path: Path to local file or directory
            repo_id: Repository ID (e.g., 'username/my-dataset')
            repo_type: Type of repository ('dataset' or 'model')
            private: Whether the repository should be private

        Returns:
            Upload result with URL
        """
        if not self.hf_token:
            return {"success": False, "error": "HUGGINGFACE_API_TOKEN not configured"}

        try:
            local_path_obj = Path(local_path)
            if not local_path_obj.exists():
                return {"success": False, "error": f"Path not found: {local_path_obj}"}

            # Create repository if it doesn't exist
            try:
                self.hf_api.create_repo(
                    repo_id=repo_id, repo_type=repo_type, private=private, exist_ok=True
                )
            except Exception:
                pass  # Repository might already exist

            # Upload
            if local_path_obj.is_file():
                url = self.hf_api.upload_file(
                    path_or_fileobj=str(local_path_obj),
                    path_in_repo=local_path_obj.name,
                    repo_id=repo_id,
                    repo_type=repo_type,
                )
            else:
                url = self.hf_api.upload_folder(
                    folder_path=str(local_path_obj), repo_id=repo_id, repo_type=repo_type
                )

            return {
                "success": True,
                "url": f"https://huggingface.co/{repo_type}s/{repo_id}",
                "uploaded": str(local_path_obj),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
