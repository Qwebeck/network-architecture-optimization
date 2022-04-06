import pygad as pg


class OnGenerationCallback:
    def __init__(self) -> None:
        self.last_fitness = 0

    def on_generation(self, ga_instance: pg.GA):
        print("Generation = {generation}".format(generation=ga_instance.generations_completed))
        print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution(
            pop_fitness=ga_instance.last_generation_fitness)[1]))
        print("Change     = {change}".format(change=ga_instance.best_solution(
            pop_fitness=ga_instance.last_generation_fitness)[1] - self.last_fitness))
        self.last_fitness = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]
