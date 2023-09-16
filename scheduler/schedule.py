from scheduler.stage import EndStage
class Schedule():
    def __init__(self, stages_ending, stages):
        self.stages_ending = stages_ending
        self.stages = stages
        assert len(stages_ending) == len(stages), "Wrong stages configuration."
    
    def get_stage(self, iter):
        for i in range(len(self.stages)):
            if iter < self.stages_ending[i]:
                return self.stages[i][iter % len(self.stages[i])]
        return EndStage()
        