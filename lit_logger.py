from neptune.new import Run


class LitLogger:
    def __init__(self, run: Run, prefix: str = ""):
        self.run = run
        self.prefix = prefix

    def log(self, label, value, step):
        self.run[f"{self.prefix}/{label}"].log(value, step=step)
