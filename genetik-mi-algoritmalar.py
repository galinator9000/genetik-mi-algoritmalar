import random, gym
import numpy as np
from deap import algorithms, base, creator, tools

## Hiperparametreler
# Her bireyin gen uzunluğu
n_features = 100

# Simüle edilecek nesil sayısı
n_generation = 100

# Popülasyon boyutu (birey sayısı)
n_population = 64

# Seleksiyon turnuvasındaki birey sayısı
selectionTournamentSize = 3

# Çaprazlama ve mutasyon olasılıkları
crossingoverProbability = 0.50
mutationProbability = 0.02

# Uygunluk ve Birey sınıflarını oluştur
creator.create("Fitness", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.Fitness)

# Bireylerin genotiplerini? ve popülasyonun basitçe birey listesi olduğunu tanımla
toolbox = base.Toolbox()
toolbox.register("attr_individual", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_individual, n_features)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Uygunluk fonksiyonunu tanımla (tuple)
def evaluate(individual):
	return (sum(individual), )

# Uygunluk fonksiyonunu kaydet
toolbox.register("evaluate", evaluate)

# Hangi çaprazlama, mutasyon ve seçilim yöntemlerinin kullanılacağını tanımla
toolbox.register("mate", tools.cxUniform, indpb=0.50)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.10)
toolbox.register("select", tools.selTournament, tournsize=selectionTournamentSize)

# Popülasyonu oluştur
population = toolbox.population(n=n_population)
stats = tools.Statistics(lambda ind: ind.fitness.values[0])
hallOfFame = tools.HallOfFame(1)

# Simülasyon başlasın!
finalPopulation, logs = algorithms.eaSimple(
	population=population,
	toolbox=toolbox,
	halloffame=hallOfFame,
	stats=stats,
	ngen=n_generation,
	cxpb=crossingoverProbability,
	mutpb=mutationProbability,
	verbose=True
)

bestSoFar = hallOfFame[0]
print("[+] En iyi uygunluk değeri: {}".format(bestSoFar.fitness.values[0]))
print("Genotip: ", bestSoFar)
