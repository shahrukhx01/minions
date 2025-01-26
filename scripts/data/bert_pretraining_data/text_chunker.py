import ray
from loguru import logger


@ray.remote
class TextChunker:
    """Chunks the text into smaller pieces of max_chars length while preserving sentence boundaries"""

    def chunk_text(self, row: dict[str, str], max_chars: int) -> list[dict[str, str]]:
        """Chunks the text into smaller pieces of max_chars length while preserving sentence boundaries

        Args:
            row (dict[str, str]): The row containing the text to be chunked
            max_chars (int): The maximum number of characters in each chunk

        Returns:
            list[dict[str, str]]: A list of dictionaries containing the chunked text
        """
        text = row["text"]
        try:
            chunks = []
            # Common sentence endings
            sentence_endings = [". ", "! ", "? ", ".\n", "!\n", "?\n"]
            current_pos = 0

            while current_pos < len(text):
                # if remaining text is shorter than max_chars, add it as the last chunk
                if current_pos + max_chars >= len(text):
                    chunks.append(text[current_pos:])
                    break

                # find the cutoff point
                cutoff = current_pos + max_chars

                # Look for the last sentence ending within the allowed length
                last_ending = -1
                for ending in sentence_endings:
                    pos = text.rfind(ending, current_pos, cutoff)
                    if pos > last_ending:
                        last_ending = pos + len(ending)

                # If no sentence ending found, look for the last space
                if last_ending == -1:
                    last_space = text.rfind(" ", current_pos, cutoff)
                    last_ending = last_space + 1 if last_space != -1 else cutoff
                if chunk := text[current_pos:last_ending].strip():
                    chunks.append(chunk)
                current_pos = last_ending
        except Exception as e:
            logger.error(f"Error chunking text: {str(e)}")
            chunks = [text]
        return [{"text": chunk} for chunk in chunks]
