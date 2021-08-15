import random
import numpy as np
from deap import algorithms, base, creator, tools

# Tekrarlanabilirlik için seed ayarlanır
SEED_VALUE = 64
random.seed(SEED_VALUE)
np.random.seed(SEED_VALUE)

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
mutationProbability = 0.05

# Uygunluk ve Birey sınıflarını oluştur
creator.create("Fitness", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.Fitness)

# Bireylerin genotiplerini ve popülasyonun basitçe birey listesi olduğunu tanımla
toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n_features)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Uygunluk fonksiyonunu tanımla
def calculateFitness(individual):
	fitness = sum(individual)
	# tuple tipinde çevirmeli
	return (fitness, )

# Uygunluk fonksiyonunu kaydet
toolbox.register("evaluate", calculateFitness)

# Hangi çaprazlama, mutasyon ve seçilim yöntemlerinin kullanılacağını tanımla
toolbox.register("mate", tools.cxOnePoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=selectionTournamentSize)

# Popülasyonu oluştur
population = toolbox.population(n=n_population)
hallOfFame = tools.HallOfFame(1)

# Fitness istatistiği
stats = tools.Statistics(lambda ind: ind.fitness.values[0])
stats.register("avg", np.mean, axis=0)
stats.register("std", np.std, axis=0)
stats.register("min", np.min, axis=0)
stats.register("max", np.max, axis=0)

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

bestIndividual = hallOfFame[0]
print("[+] En iyi uygunluk değeri: {}".format(bestIndividual.fitness.values[0]))
print("Genotip: ", bestIndividual)
