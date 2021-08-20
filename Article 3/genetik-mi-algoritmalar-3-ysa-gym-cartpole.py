import random, gym
import numpy as np
from deap import algorithms, base, creator, tools
from common import NeuralNetwork, run_gym_environment

# Simülasyon ortamımızı oluşturalım
env = gym.make("CartPole-v1")
observation_space_dim = env.observation_space.shape[0]
action_space_dim = env.action_space.n

# Tekrarlanabilirlik için seed ayarlayalım
SEED_VALUE = 64
random.seed(SEED_VALUE)
np.random.seed(SEED_VALUE)
env.seed(SEED_VALUE)

## Hiperparametreler
# Yapay sinir ağı girdi-çıktı arasındaki gizli katmanları (nöron sayısı)
nn_hidden_layer_units = [2]

# Simüle edilecek nesil sayısı
n_generation = 20

# Popülasyon boyutu (birey sayısı)
n_population = 32

# Seleksiyon turnuvasında rastgele seçilecek birey sayısı (k)
selectionTournamentSize = 3

# Birey bazlı mutasyon olasılığı
individualMutationProbability = 0.05

# Gen bazlı mutasyon olasılığı
geneMutationProbability = 0.05

# Çaprazlama olasılığı
crossoverProbability = 0.80

# Yapay sinir ağı modelimiz
nn = NeuralNetwork(
	layer_units=[
		# Girdi katman nöron sayısı
		observation_space_dim,

		# Gizli katmanlar
		# başına * koyarak liste elemanlarını, liste tanımı içine açabiliyoruz
		*nn_hidden_layer_units,

		# Çıktı katman nöron sayısı
		action_space_dim
	]
)

## Genetik algoritmalar
# Uygunluk ve Birey sınıflarını oluştur
creator.create("Fitness", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.Fitness)

# Bireylerin genotiplerini ve popülasyonun basitçe birey listesi olduğunu tanımla
toolbox = base.Toolbox()
toolbox.register("initializeGene", random.gauss, 0, 1)
toolbox.register("initializeIndividual", tools.initRepeat, creator.Individual, toolbox.initializeGene, n=nn.get_parameter_count())
toolbox.register("initializePopulation", tools.initRepeat, list, toolbox.initializeIndividual, n=n_population)

# Uygunluk fonksiyonumuzu tanımlayalım: yapay sinir ağının, ortamdan ne kadar ödül toplayabildiği. Hepsi bu.
def calculateFitness(individual):
	individualEpisodeRewards = run_gym_environment(
		env=env,
		# Aksiyon fonksiyonumuz, bireyin genotipini kullanarak yapay sinir ağını çalıştırır
		act_fn=(lambda state: nn.forward(individual, state))
	)
	# tuple tipinde çevirmeli
	return (individualEpisodeRewards, )

# Uygunluk fonksiyonunu kaydet
toolbox.register("evaluate", calculateFitness)

# Hangi çaprazlama, mutasyon ve seçilim yöntemlerinin kullanılacağını tanımla
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=geneMutationProbability)
toolbox.register("select", tools.selTournament, tournsize=selectionTournamentSize)

# Popülasyonu oluştur
initialPopulation = toolbox.initializePopulation(n=n_population)
# En iyi bireyin kaydını tutacak obje
hallOfFame = tools.HallOfFame(1)

# Uygunluk istatistiklerini tutacak objemiz
stats = tools.Statistics(lambda ind: ind.fitness.values[0])
stats.register("avg", np.mean, axis=0)
stats.register("std", np.std, axis=0)
stats.register("min", np.min, axis=0)
stats.register("max", np.max, axis=0)

# Evrimsel simülasyon öncesi başlangıç popülasyonunun her bireyiyle ortamı 1 episode çalıştır
print("[*] Evrimsel simülasyon öncesi popülasyon")
for ind in initialPopulation:
	run_gym_environment(
		env=env,
		act_fn=(
			lambda state: nn.forward(
				ind,
				state
			)
		),
		n_episode=1,
		render=True,
		n_timestep=40
	)

# Evrimsel simülasyon başlasın!
finalPopulation, logs = algorithms.eaSimple(
	population=initialPopulation,
	toolbox=toolbox,
	halloffame=hallOfFame,
	stats=stats,
	ngen=n_generation,
	cxpb=crossoverProbability,
	mutpb=individualMutationProbability,
	verbose=True
)
topIndividual = hallOfFame[0]

# En iyi bireyle simülasyonu çalıştır
print("\n[+] En iyi uygunluk sağlayan bireyin skoru: {}".format(topIndividual.fitness.values[0]))
print("Genotipi: ", topIndividual)

run_gym_environment(
	env=env,
	act_fn=(lambda state: nn.forward(topIndividual, state)),
	n_episode=5,
	render=True
)
env.close()
