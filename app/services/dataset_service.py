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

    def __init__(self):
        self.kaggle_token = os.environ.get("KAGGLE_API_TOKEN")
        self.hf_token = os.environ.get("HUGGINGFACE_API_TOKEN")
        self.data_dir = Path(os.environ.get("DATA_DIR", "data"))
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
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
        self,
        query: str = "",
        task: Optional[str] = None,
        limit: int = 20,
        sort: str = "downloads"
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
            datasets = list(list_datasets(
                search=query if query else None,
                task_categories=task,
                sort=sort,
                direction=-1,  # Descending
                limit=limit
            ))
            
            results = []
            for ds in datasets:
                results.append({
                    "id": ds.id,
                    "author": ds.author,
                    "name": ds.id.split("/")[-1] if "/" in ds.id else ds.id,
                    "downloads": getattr(ds, "downloads", 0),
                    "likes": getattr(ds, "likes", 0),
                    "tags": getattr(ds, "tags", []),
                    "created_at": str(getattr(ds, "created_at", "")),
                    "source": "huggingface",
                    "url": f"https://huggingface.co/datasets/{ds.id}"
                })
            
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
                "url": f"https://huggingface.co/datasets/{dataset_id}"
            }
        except Exception as e:
            return {"error": str(e), "id": dataset_id}

    def download_hf_dataset(
        self,
        dataset_id: str,
        subset: Optional[str] = None,
        split: str = "train"
    ) -> Dict[str, Any]:
        """Download a HuggingFace dataset.
        
        Args:
            dataset_id: Dataset identifier
            subset: Dataset subset/configuration (e.g., 'sentiment' for tweet_eval)
            split: Dataset split to download
            
        Returns:
            Download result with file path and stats
        """
        try:
            from datasets import load_dataset
            
            # Load dataset
            if subset:
                dataset = load_dataset(dataset_id, subset, split=split)
            else:
                dataset = load_dataset(dataset_id, split=split)
            
            # Save to parquet
            output_dir = self.data_dir / "raw" / "hf" / dataset_id.replace("/", "_")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            filename = f"{subset}_{split}" if subset else split
            output_path = output_dir / f"{filename}.parquet"
            
            df = pd.DataFrame(dataset)
            df.to_parquet(output_path, index=False)
            
            return {
                "success": True,
                "dataset_id": dataset_id,
                "subset": subset,
                "split": split,
                "path": str(output_path),
                "num_rows": len(df),
                "columns": list(df.columns),
                "size_mb": output_path.stat().st_size / 1024 / 1024
            }
        except Exception as e:
            return {"success": False, "error": str(e), "dataset_id": dataset_id}

    # =========================================================================
    # Kaggle Methods
    # =========================================================================

    def search_kaggle_datasets(
        self,
        query: str = "",
        sort_by: str = "hottest",
        file_type: str = "csv",
        limit: int = 20
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
                ["kaggle", "datasets", "list", "-s", query, "--sort-by", sort_by, 
                 "--file-type", file_type, "-v", "--csv"],
                capture_output=True,
                text=True,
                timeout=30
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
                results.append({
                    "id": row.get("ref", ""),
                    "name": row.get("title", row.get("ref", "").split("/")[-1]),
                    "author": row.get("ref", "").split("/")[0] if "/" in row.get("ref", "") else "",
                    "size": row.get("size", ""),
                    "downloads": int(row.get("downloadCount", 0)),
                    "votes": int(row.get("voteCount", 0)),
                    "last_updated": row.get("lastUpdated", ""),
                    "source": "kaggle",
                    "url": f"https://www.kaggle.com/datasets/{row.get('ref', '')}"
                })
            
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
                timeout=30
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
                    "url": f"https://www.kaggle.com/datasets/{dataset_id}"
                }
            
            return {"id": dataset_id, "source": "kaggle"}
        except Exception as e:
            return {"error": str(e), "id": dataset_id}

    def download_kaggle_dataset(
        self,
        dataset_id: str,
        unzip: bool = True
    ) -> Dict[str, Any]:
        """Download a Kaggle dataset.
        
        Args:
            dataset_id: Dataset identifier (e.g., 'kazanova/sentiment140')
            unzip: Whether to unzip the downloaded files
            
        Returns:
            Download result with file path and stats
        """
        try:
            output_dir = self.data_dir / "raw" / "kaggle" / dataset_id.replace("/", "_")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            cmd = ["kaggle", "datasets", "download", "-d", dataset_id, 
                   "-p", str(output_dir)]
            if unzip:
                cmd.append("--unzip")
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                return {"success": False, "error": result.stderr, "dataset_id": dataset_id}
            
            # Get downloaded files info
            files = []
            total_size = 0
            for f in output_dir.rglob("*"):
                if f.is_file():
                    size = f.stat().st_size
                    total_size += size
                    files.append({
                        "name": f.name,
                        "path": str(f),
                        "size_mb": size / 1024 / 1024
                    })
            
            return {
                "success": True,
                "dataset_id": dataset_id,
                "path": str(output_dir),
                "files": files,
                "total_size_mb": total_size / 1024 / 1024
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Download timeout", "dataset_id": dataset_id}
        except Exception as e:
            return {"success": False, "error": str(e), "dataset_id": dataset_id}

    # =========================================================================
    # Local Dataset Management
    # =========================================================================

    def list_local_datasets(self) -> List[Dict[str, Any]]:
        """List all locally downloaded datasets.
        
        Returns:
            List of local dataset information
        """
        datasets = []
        
        for source_dir in ["raw", "processed"]:
            source_path = self.data_dir / source_dir
            if not source_path.exists():
                continue
                
            for dataset_dir in source_path.rglob("*"):
                if dataset_dir.is_dir():
                    files = list(dataset_dir.glob("*"))
                    if files:
                        total_size = sum(f.stat().st_size for f in files if f.is_file())
                        datasets.append({
                            "name": dataset_dir.name,
                            "path": str(dataset_dir),
                            "type": source_dir,
                            "num_files": len([f for f in files if f.is_file()]),
                            "size_mb": total_size / 1024 / 1024,
                            "files": [f.name for f in files if f.is_file()][:10]
                        })
        
        return datasets

    def get_dataset_preview(
        self,
        path: str,
        num_rows: int = 10
    ) -> Dict[str, Any]:
        """Get a preview of a dataset file.
        
        Args:
            path: Path to the dataset file
            num_rows: Number of rows to preview
            
        Returns:
            Preview data with sample rows and statistics
        """
        path = Path(path)
        if not path.exists():
            return {"error": f"File not found: {path}"}
        
        try:
            if path.suffix == ".parquet":
                df = pd.read_parquet(path)
            elif path.suffix == ".csv":
                df = pd.read_csv(path, nrows=num_rows * 10)
            elif path.suffix == ".json":
                df = pd.read_json(path, lines=True, nrows=num_rows * 10)
            else:
                return {"error": f"Unsupported file format: {path.suffix}"}
            
            # Get statistics
            stats = {
                "total_rows": len(df),
                "columns": list(df.columns),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "memory_mb": df.memory_usage(deep=True).sum() / 1024 / 1024
            }
            
            # Get value counts for categorical columns
            for col in df.columns:
                if df[col].dtype == "object" or df[col].nunique() < 20:
                    stats[f"{col}_distribution"] = df[col].value_counts().head(10).to_dict()
            
            return {
                "path": str(path),
                "preview": df.head(num_rows).to_dict(orient="records"),
                "stats": stats
            }
        except Exception as e:
            return {"error": str(e), "path": str(path)}

    def delete_dataset(self, path: str) -> Dict[str, Any]:
        """Delete a local dataset.
        
        Args:
            path: Path to the dataset directory
            
        Returns:
            Deletion result
        """
        path = Path(path)
        if not path.exists():
            return {"success": False, "error": "Path not found"}
        
        # Safety check - only allow deletion within data directory
        try:
            path.relative_to(self.data_dir)
        except ValueError:
            return {"success": False, "error": "Cannot delete files outside data directory"}
        
        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            return {"success": True, "deleted": str(path)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # =========================================================================
    # HuggingFace Upload (for your trained models/datasets)
    # =========================================================================

    def upload_to_huggingface(
        self,
        local_path: str,
        repo_id: str,
        repo_type: str = "dataset",
        private: bool = False
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
            local_path = Path(local_path)
            if not local_path.exists():
                return {"success": False, "error": f"Path not found: {local_path}"}
            
            # Create repository if it doesn't exist
            try:
                self.hf_api.create_repo(
                    repo_id=repo_id,
                    repo_type=repo_type,
                    private=private,
                    exist_ok=True
                )
            except Exception:
                pass  # Repository might already exist
            
            # Upload
            if local_path.is_file():
                url = self.hf_api.upload_file(
                    path_or_fileobj=str(local_path),
                    path_in_repo=local_path.name,
                    repo_id=repo_id,
                    repo_type=repo_type
                )
            else:
                url = self.hf_api.upload_folder(
                    folder_path=str(local_path),
                    repo_id=repo_id,
                    repo_type=repo_type
                )
            
            return {
                "success": True,
                "url": f"https://huggingface.co/{repo_type}s/{repo_id}",
                "uploaded": str(local_path)
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
