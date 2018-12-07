class ProgressLogger:
    """
    Logging on progress.

    Attributes:
        cnt (int): how many trials have been completed
        total (int): total number of trials need to be processed
        annotation (string): annotation string at the head of progress bar
        length (int): length of progress bar
        progress_bar (string): logging bar format

    Format:
        $ annotation [===>.......] cnt/total
    """
    def __init__(self, total, annotation=None, length=None):
        assert total > 0, "negative total number of trials"
        self.cnt = 0
        self.total = total
        self.annotation = annotation if annotation is not None else ""
        self.length = length if length is not None else 50
        self.progress_bar = "[" + "." * self.length + "] {}/{}"

    def log(self):
        prev_progress = int(self.cnt / self.total * self.length)
        self.cnt += 1
        progress = int(self.cnt / self.total * self.length)
        add, bias = ">", 2
        if progress >= self.length:
            add, bias = "", 1
        self.progress_bar = self.progress_bar[:prev_progress+1] + "=" * int(progress - prev_progress) + add + self.progress_bar[progress+bias:]
        if progress >= self.length:
            print((self.annotation + self.progress_bar).format(self.cnt, self.total))
        else:
            print((self.annotation + self.progress_bar).format(self.cnt, self.total), end="\r", flush=True)