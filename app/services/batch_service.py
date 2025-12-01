"""Batch processing service for CSV sentiment analysis.

This module provides functionality for processing CSV files containing
text data and performing sentiment/emotion analysis on each row.
"""

import os
import csv
import io
from datetime import datetime
from typing import Optional, Dict, Any, BinaryIO

import pandas as pd

from app.services.sentiment_service import SentimentService


class BatchService:
    """Service for batch processing CSV files for sentiment analysis.

    Handles CSV file validation, processing, and result generation
    with progress tracking support for Celery integration.

    Attributes:
        sentiment_service: SentimentService instance for analysis
        max_file_size: Maximum allowed file size in bytes
        max_rows: Maximum number of rows to process
    """

    DEFAULT_TEXT_COLUMN = 'Text'
    ALLOWED_TEXT_COLUMNS = ['text', 'Text', 'content', 'Content', 'message',
                            'Message', 'review', 'Review', 'comment', 'Comment']

    def __init__(
        self,
        sentiment_service: SentimentService = None,
        max_file_size: int = 16 * 1024 * 1024,  # 16MB
        max_rows: int = 50000
    ):
        """Initialize batch service.

        Args:
            sentiment_service: Optional SentimentService instance
            max_file_size: Maximum file size in bytes
            max_rows: Maximum rows to process
        """
        self.sentiment_service = sentiment_service or SentimentService()
        self.max_file_size = max_file_size
        self.max_rows = max_rows

    def validate_file(self, file: BinaryIO, filename: str) -> Dict[str, Any]:
        """Validate uploaded CSV file.

        Args:
            file: File-like object
            filename: Original filename

        Returns:
            dict: Validation result with keys:
                - valid: Boolean indicating if file is valid
                - error: Error message if invalid
                - row_count: Number of rows in file
                - text_column: Detected text column name
                - columns: List of all column names

        Raises:
            ValueError: If file is invalid
        """
        # Check file extension
        if not filename.lower().endswith('.csv'):
            return {
                'valid': False,
                'error': 'File must be a CSV file'
            }

        # Check file size
        file.seek(0, 2)  # Seek to end
        size = file.tell()
        file.seek(0)  # Reset to beginning

        if size > self.max_file_size:
            return {
                'valid': False,
                'error': f'File size ({size} bytes) exceeds maximum '
                        f'({self.max_file_size} bytes)'
            }

        if size == 0:
            return {
                'valid': False,
                'error': 'File is empty'
            }

        # Try to read CSV
        try:
            # Read first chunk to validate structure
            content = file.read().decode('utf-8')
            file.seek(0)

            df = pd.read_csv(io.StringIO(content), nrows=5)

            if df.empty:
                return {
                    'valid': False,
                    'error': 'CSV file contains no data'
                }

            # Find text column
            text_column = None
            for col in self.ALLOWED_TEXT_COLUMNS:
                if col in df.columns:
                    text_column = col
                    break

            if not text_column:
                return {
                    'valid': False,
                    'error': f'CSV must contain one of these columns: '
                            f'{", ".join(self.ALLOWED_TEXT_COLUMNS)}'
                }

            # Count total rows
            df_full = pd.read_csv(io.StringIO(content))
            row_count = len(df_full)

            if row_count > self.max_rows:
                return {
                    'valid': False,
                    'error': f'File contains {row_count} rows, maximum is {self.max_rows}'
                }

            return {
                'valid': True,
                'row_count': row_count,
                'text_column': text_column,
                'columns': list(df.columns)
            }

        except pd.errors.EmptyDataError:
            return {
                'valid': False,
                'error': 'CSV file is empty or malformed'
            }
        except pd.errors.ParserError as e:
            return {
                'valid': False,
                'error': f'CSV parsing error: {str(e)}'
            }
        except UnicodeDecodeError:
            return {
                'valid': False,
                'error': 'File encoding not supported. Please use UTF-8 encoded CSV.'
            }
        except Exception as e:
            return {
                'valid': False,
                'error': f'Error reading file: {str(e)}'
            }

    def process_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str,
        include_emotions: bool = True,
        batch_size: int = 32,
        progress_callback=None
    ) -> pd.DataFrame:
        """Process a DataFrame and add sentiment/emotion columns.

        Args:
            df: Input DataFrame
            text_column: Name of column containing text
            include_emotions: Whether to include emotion analysis
            batch_size: Batch size for transformer processing
            progress_callback: Optional callback(processed, total, stats)

        Returns:
            pd.DataFrame: DataFrame with added sentiment/emotion columns
        """
        texts = df[text_column].fillna('').astype(str).tolist()
        total = len(texts)

        # Track statistics
        stats = {
            'positive': 0,
            'negative': 0,
            'neutral': 0,
            'emotions': {}
        }

        def update_progress(processed, total_count):
            if progress_callback:
                progress_callback(processed, total_count, stats)

        # Perform batch analysis
        results = self.sentiment_service.batch_analyze(
            texts,
            batch_size=batch_size,
            include_emotions=include_emotions,
            progress_callback=update_progress
        )

        # Add results to DataFrame
        sentiments = []
        compound_scores = []
        pos_scores = []
        neg_scores = []
        neu_scores = []
        primary_emotions = []
        confidences = []

        for result in results:
            sentiment = result['sentiment']
            sentiments.append(sentiment)
            compound_scores.append(result['compound_score'])
            pos_scores.append(result['scores']['pos'])
            neg_scores.append(result['scores']['neg'])
            neu_scores.append(result['scores']['neu'])

            # Update sentiment stats
            stats[sentiment] = stats.get(sentiment, 0) + 1

            if include_emotions and 'primary_emotion' in result:
                emotion = result['primary_emotion']
                primary_emotions.append(emotion)
                confidences.append(result['confidence'])

                # Update emotion stats
                stats['emotions'][emotion] = stats['emotions'].get(emotion, 0) + 1
            else:
                primary_emotions.append(None)
                confidences.append(None)

        # Add columns to DataFrame
        df = df.copy()
        df['Sentiment'] = sentiments
        df['Compound_Score'] = compound_scores
        df['Positive_Score'] = pos_scores
        df['Negative_Score'] = neg_scores
        df['Neutral_Score'] = neu_scores

        if include_emotions:
            df['Primary_Emotion'] = primary_emotions
            df['Emotion_Confidence'] = confidences

        # Final progress update
        if progress_callback:
            progress_callback(total, total, stats)

        return df

    def process_file(
        self,
        file_path: str,
        output_path: str = None,
        include_emotions: bool = True,
        batch_size: int = 32,
        progress_callback=None
    ) -> Dict[str, Any]:
        """Process a CSV file and save results.

        Args:
            file_path: Path to input CSV file
            output_path: Path for output CSV (optional)
            include_emotions: Whether to include emotion analysis
            batch_size: Batch size for transformer processing
            progress_callback: Optional callback(processed, total, stats)

        Returns:
            dict: Processing result with keys:
                - success: Boolean indicating success
                - output_path: Path to output file
                - row_count: Number of rows processed
                - stats: Sentiment/emotion statistics
                - error: Error message if failed
        """
        try:
            # Read CSV
            df = pd.read_csv(file_path)

            # Find text column
            text_column = None
            for col in self.ALLOWED_TEXT_COLUMNS:
                if col in df.columns:
                    text_column = col
                    break

            if not text_column:
                return {
                    'success': False,
                    'error': 'Text column not found in CSV'
                }

            # Process DataFrame
            result_df = self.process_dataframe(
                df,
                text_column,
                include_emotions=include_emotions,
                batch_size=batch_size,
                progress_callback=progress_callback
            )

            # Generate output path if not provided
            if not output_path:
                base, ext = os.path.splitext(file_path)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_path = f"{base}_analyzed_{timestamp}{ext}"

            # Save results
            result_df.to_csv(output_path, index=False)

            # Calculate final statistics
            stats = {
                'positive': int((result_df['Sentiment'] == 'positive').sum()),
                'negative': int((result_df['Sentiment'] == 'negative').sum()),
                'neutral': int((result_df['Sentiment'] == 'neutral').sum())
            }

            if include_emotions and 'Primary_Emotion' in result_df.columns:
                emotion_counts = result_df['Primary_Emotion'].value_counts()
                stats['emotions'] = emotion_counts.to_dict()

            return {
                'success': True,
                'output_path': output_path,
                'row_count': len(result_df),
                'stats': stats
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def generate_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate a summary of analysis results.

        Args:
            df: Analyzed DataFrame with sentiment columns

        Returns:
            dict: Summary statistics
        """
        summary = {
            'total_records': len(df),
            'sentiment_distribution': {},
            'emotion_distribution': {},
            'average_scores': {}
        }

        # Sentiment distribution
        if 'Sentiment' in df.columns:
            sentiment_counts = df['Sentiment'].value_counts()
            total = len(df)
            summary['sentiment_distribution'] = {
                'positive': {
                    'count': int(sentiment_counts.get('positive', 0)),
                    'percentage': round(sentiment_counts.get('positive', 0) / total * 100, 2)
                },
                'negative': {
                    'count': int(sentiment_counts.get('negative', 0)),
                    'percentage': round(sentiment_counts.get('negative', 0) / total * 100, 2)
                },
                'neutral': {
                    'count': int(sentiment_counts.get('neutral', 0)),
                    'percentage': round(sentiment_counts.get('neutral', 0) / total * 100, 2)
                }
            }

        # Emotion distribution
        if 'Primary_Emotion' in df.columns:
            emotion_counts = df['Primary_Emotion'].value_counts()
            total = len(df)
            summary['emotion_distribution'] = {
                emotion: {
                    'count': int(count),
                    'percentage': round(count / total * 100, 2)
                }
                for emotion, count in emotion_counts.items()
            }

        # Average scores
        score_columns = ['Compound_Score', 'Positive_Score', 'Negative_Score',
                        'Neutral_Score', 'Emotion_Confidence']
        for col in score_columns:
            if col in df.columns:
                summary['average_scores'][col] = round(df[col].mean(), 4)

        return summary

    @staticmethod
    def dataframe_to_csv_string(df: pd.DataFrame) -> str:
        """Convert DataFrame to CSV string for download.

        Args:
            df: DataFrame to convert

        Returns:
            str: CSV formatted string
        """
        output = io.StringIO()
        df.to_csv(output, index=False)
        return output.getvalue()
