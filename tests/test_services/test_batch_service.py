"""Tests for batch processing service."""

import pytest
import io
import pandas as pd
from app.services.batch_service import BatchService


class TestBatchService:
    """Test cases for BatchService."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixtures."""
        self.service = BatchService()

    def test_validate_file_valid_csv(self, tmp_path):
        """Test validation accepts valid CSV file."""
        csv_content = "Text\nI love this!\nThis is bad.\n"
        csv_file = io.BytesIO(csv_content.encode("utf-8"))

        result = self.service.validate_file(csv_file, "test.csv")

        assert result["valid"] is True
        assert result["row_count"] == 2
        assert result["text_column"] == "Text"
        assert "Text" in result["columns"]

    def test_validate_file_alternative_column_names(self, tmp_path):
        """Test validation accepts alternative column names."""
        for col_name in ["text", "content", "message", "review"]:
            csv_content = f"{col_name}\nTest text\n"
            csv_file = io.BytesIO(csv_content.encode("utf-8"))

            result = self.service.validate_file(csv_file, "test.csv")

            assert result["valid"] is True
            assert result["text_column"] == col_name

    def test_validate_file_missing_text_column(self):
        """Test validation rejects CSV without text column."""
        csv_content = "id,name,value\n1,test,123\n"
        csv_file = io.BytesIO(csv_content.encode("utf-8"))

        result = self.service.validate_file(csv_file, "test.csv")

        assert result["valid"] is False
        assert "column" in result["error"].lower()

    def test_validate_file_wrong_extension(self):
        """Test validation rejects non-CSV files."""
        result = self.service.validate_file(io.BytesIO(b""), "test.txt")

        assert result["valid"] is False
        assert "CSV" in result["error"]

    def test_validate_file_empty_file(self):
        """Test validation rejects empty file."""
        csv_file = io.BytesIO(b"")

        result = self.service.validate_file(csv_file, "test.csv")

        assert result["valid"] is False
        assert "empty" in result["error"].lower()

    def test_validate_file_too_large(self):
        """Test validation rejects oversized files."""
        # Create a service with a small max size
        service = BatchService(max_file_size=100)

        csv_content = "Text\n" + "x" * 200
        csv_file = io.BytesIO(csv_content.encode("utf-8"))

        result = service.validate_file(csv_file, "test.csv")

        assert result["valid"] is False
        assert "size" in result["error"].lower()

    def test_validate_file_too_many_rows(self):
        """Test validation rejects files with too many rows."""
        service = BatchService(max_rows=5)

        csv_content = "Text\n" + "Test\n" * 10
        csv_file = io.BytesIO(csv_content.encode("utf-8"))

        result = service.validate_file(csv_file, "test.csv")

        assert result["valid"] is False
        assert "rows" in result["error"].lower()

    def test_process_dataframe_adds_sentiment_columns(self):
        """Test processing adds sentiment columns."""
        df = pd.DataFrame({"Text": ["I love this!", "I hate this.", "It is okay."]})

        result_df = self.service.process_dataframe(df, "Text", include_emotions=False)

        assert "Sentiment" in result_df.columns
        assert "Compound_Score" in result_df.columns
        assert "Positive_Score" in result_df.columns
        assert "Negative_Score" in result_df.columns
        assert "Neutral_Score" in result_df.columns

    def test_process_dataframe_adds_emotion_columns(self):
        """Test processing adds emotion columns when requested."""
        df = pd.DataFrame({"Text": ["I am so happy!", "I am very sad."]})

        result_df = self.service.process_dataframe(df, "Text", include_emotions=True)

        assert "Primary_Emotion" in result_df.columns
        assert "Emotion_Confidence" in result_df.columns

    def test_process_dataframe_preserves_original_columns(self):
        """Test processing preserves original columns."""
        df = pd.DataFrame({"id": [1, 2, 3], "Text": ["A", "B", "C"], "extra": ["x", "y", "z"]})

        result_df = self.service.process_dataframe(df, "Text", include_emotions=False)

        assert "id" in result_df.columns
        assert "extra" in result_df.columns
        assert list(result_df["id"]) == [1, 2, 3]

    def test_process_dataframe_handles_empty_text(self):
        """Test processing handles empty or null text values."""
        df = pd.DataFrame({"Text": ["Good", "", None, "Bad"]})

        result_df = self.service.process_dataframe(df, "Text", include_emotions=False)

        assert len(result_df) == 4
        assert all(result_df["Sentiment"].notna())

    def test_process_dataframe_progress_callback(self):
        """Test processing calls progress callback."""
        df = pd.DataFrame({"Text": ["Text " + str(i) for i in range(10)]})

        progress_updates = []

        def callback(processed, total, stats):
            progress_updates.append((processed, total, stats.copy()))

        self.service.process_dataframe(
            df, "Text", include_emotions=False, progress_callback=callback
        )

        assert len(progress_updates) > 0
        last_update = progress_updates[-1]
        assert last_update[0] == len(df)
        assert last_update[1] == len(df)

    def test_process_file_creates_output(self, tmp_path):
        """Test processing creates output file."""
        # Create input file
        input_path = tmp_path / "input.csv"
        df = pd.DataFrame({"Text": ["I love it!", "I hate it."]})
        df.to_csv(input_path, index=False)

        output_path = tmp_path / "output.csv"

        result = self.service.process_file(
            str(input_path), str(output_path), include_emotions=False
        )

        assert result["success"] is True
        assert result["row_count"] == 2
        assert output_path.exists()

        # Verify output content
        output_df = pd.read_csv(output_path)
        assert "Sentiment" in output_df.columns

    def test_process_file_returns_stats(self, tmp_path):
        """Test processing returns statistics."""
        input_path = tmp_path / "input.csv"
        df = pd.DataFrame({"Text": ["Great!", "Terrible!", "Okay."]})
        df.to_csv(input_path, index=False)

        result = self.service.process_file(str(input_path), include_emotions=False)

        assert result["success"] is True
        assert "stats" in result
        assert "positive" in result["stats"]
        assert "negative" in result["stats"]
        assert "neutral" in result["stats"]

    def test_process_file_handles_missing_file(self):
        """Test processing handles missing input file."""
        result = self.service.process_file("/nonexistent/file.csv")

        assert result["success"] is False
        assert "error" in result

    def test_generate_summary(self):
        """Test summary generation."""
        df = pd.DataFrame(
            {
                "Sentiment": ["positive", "positive", "negative", "neutral"],
                "Compound_Score": [0.8, 0.6, -0.7, 0.0],
                "Primary_Emotion": ["joy", "joy", "anger", "neutral"],
                "Emotion_Confidence": [0.9, 0.8, 0.7, 0.6],
            }
        )

        summary = self.service.generate_summary(df)

        assert summary["total_records"] == 4
        assert summary["sentiment_distribution"]["positive"]["count"] == 2
        assert summary["sentiment_distribution"]["negative"]["count"] == 1
        assert summary["sentiment_distribution"]["neutral"]["count"] == 1
        assert "joy" in summary["emotion_distribution"]
        assert summary["emotion_distribution"]["joy"]["count"] == 2

    def test_dataframe_to_csv_string(self):
        """Test DataFrame to CSV string conversion."""
        df = pd.DataFrame({"A": [1, 2], "B": ["x", "y"]})

        csv_string = BatchService.dataframe_to_csv_string(df)

        assert "A,B" in csv_string
        assert "1,x" in csv_string
        assert "2,y" in csv_string
