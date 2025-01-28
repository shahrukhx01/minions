import ray
from langdetect import detect
from loguru import logger


@ray.remote
class LanguageDetector:
    """Detects the language of the text"""

    def detect_language(self, row: dict[str, str]) -> dict[str, str]:
        """Detects the language of the text using the langdetect library

        Args:
            row (dict[str, str]): The row containing the text to be detected

        Returns:
            dict[str, str]: The row with the detected language added
        """
        text = row["text"]
        try:
            row["language"] = detect(text)
        except Exception as e:
            logger.error(f"Error detecting language: {str(e)}")
            row["language"] = "xx"
        return row
