import matplotlib.pyplot as plt

from typing import (
    List,
)


class PlotHandler:
    def __init__(self, loss: List[float] = [],
                 record: int = 0,
                 scores: List[int] = [],
                 predicted_ratio: List[float] = []) -> None:
        self.scores = scores
        self.records: List[int] = [record]
        self.record_games: List[int] = [0]
        self.predicted_ratio = predicted_ratio
        self.total_score = 0

        self.loss = loss
        #self.max_loss = np.finfo(float).tiny

        self.fig, self.axs = plt.subplots(2, 1)
        self.ax2 = self.axs[0].twinx()
        plt.ion()

        self.ma: List[float] = []
        self.mean: List[float] = []
        self.mean_diff: List[float] = []

    def moving_avg(self, x: List, n: int) -> None:
        self.ma.append(sum(x[-n:]) / n)

    def _scores_calc(self, avg: int, games: int,  loss: float, record: int, score: int, predicted_ratio: float) -> None:
        # if loss > self.max_loss:
        #     self.max_loss = loss
        # self.axs[1].plot(self.loss, color='red')
        # self.axs[1].set_ylabel("Loss")
        if record > self.records[-1]:
            self.records.append(record)
            self.record_games.append(games - 1)
        self.total_score += score
        self.scores.append(score)
        self.predicted_ratio.append(predicted_ratio)
        self.mean.append(self.total_score / games)
        self.moving_avg(self.scores, avg)
        self.mean_diff.append(self.ma[-1] - self.mean[-1])
        self.loss.append(loss)

    def plot(self, avg: int, games: int,  loss: float, record: int, score: int, predicted_ratio: float) -> None:
        self._scores_calc(avg, games,  loss, record, score, predicted_ratio)
        self.axs[0].plot(self.scores, color='red')
        self.axs[0].scatter(self.record_games, self.records, color='red')
        self.axs[0].plot(self.mean, color='green')
        self.axs[0].plot(self.ma, color='blue')
        self.axs[0].set_ylabel("Game Score")
        self.axs[0].legend(
            ['Score', f'Record: {record}', 'Score mean',
                f'Score avg. {avg} games'],
            loc='upper left')
        self.ax2.plot(self.predicted_ratio, ls='--', color='orange')
        self.ax2.set_ylabel('steps predicted / total steps', color='orange')
        self.axs[1].plot(self.mean_diff, color='red')
        self.axs[1].set_ylabel(f'Avg {avg} games - Score mean')
        self.axs[1].set_xlabel("Games Played")
        self.fig.show()
        plt.pause(1)
