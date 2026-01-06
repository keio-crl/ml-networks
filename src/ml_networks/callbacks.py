"""コールバックを扱うモジュール."""

from __future__ import annotations

from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme


class ProgressBarCallback(RichProgressBar):
    """
    Make the progress bar richer.

    References
    ----------
    * https://qiita.com/akihironitta/items/edfd6b29dfb67b17fb00
    """

    def __init__(self) -> None:
        """Rich progress bar with custom theme."""
        theme = RichProgressBarTheme(
            description="green_yellow",
            progress_bar="green1",
            progress_bar_finished="green1",
            batch_progress="green_yellow",
            time="grey82",
            processing_speed="grey82",
            metrics="grey82",
        )
        super().__init__(theme=theme)
