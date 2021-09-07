import random, gym, pickle, concurrent.futures
import numpy as np
from deap import algorithms, base, creator, tools

trainingSession = 2
modelOutPath = "models/{}/bipedalWalkerBest".format(trainingSession)
loadPopulation = False
initialPopulationInPath = "models{}//initialPopulation.pickle".format(trainingSession)
allGenerationsFinalPopulationOutPath = "models/{}/allGenerationsFinalPopulation.pickle".format(trainingSession)
finalPopulationOutPath = "models/{}/finalPopulation.pickle".format(trainingSession)

## Simülasyon ortamı
GYM_ENV_NAME = "BipedalWalker-v3"
env = gym.make(GYM_ENV_NAME)
observation_space_dim = env.observation_space.shape[0]
action_space_dim = env.action_space.shape[0]
env.close()

# Tekrarlanabilirlik için seed ayarlanır
SEED_VALUE = 256
random.seed(SEED_VALUE)
np.random.seed(SEED_VALUE)

# Simülasyonu çalıştırmak için ortak fonksiyon
def run_env(act_fn, n_episode=1, render=False, max_timestep=None):
	env = gym.make(GYM_ENV_NAME)
	env.seed(SEED_VALUE)

	# Episode döngüsü
	totalRewards = 0
	for episode in range(n_episode):
		state = env.reset()
		done = False

		# Timestep döngüsü
		timestep = 0
		while not done:
			if (not max_timestep == None) and (timestep >= max_timestep):
				break
			if render: env.render()

			# act_fn adlı fonksiyon parametreyi çağırarak aksiyonu al
			action = act_fn(state)
			next_state, reward, done, info = env.step(action)

			totalRewards += reward
			state = next_state
			timestep += 1

	env.close()
	return totalRewards

## Yapay sinir ağı
# Ara katman ünite sayısı
nn_hidden_unit = 4
w1_ndim = (observation_space_dim*nn_hidden_unit)
w2_ndim = (nn_hidden_unit*action_space_dim)

# Verilen bireyin genotipini yapay sinir ağı parametreleri olarak kullanarak, state vektörünü feed-forward eder
def nn_forward(individual, state):
	# State'in 2 seviyeli array olduğundan emin ol: [?, observation_space_dim]
	if len(state.shape) == 1: state = np.array([state])
	assert len(state.shape) == 2

	# Yapay sinir ağı parametrelerini genotip olan 1D vektörden çıkar
	arr = np.array(individual)
	w1 = arr[:w1_ndim]
	b1 = arr[w1_ndim]
	w2 = arr[w1_ndim+1 : w1_ndim+1+w2_ndim]
	b2 = arr[-1]

	# Ağırlıkları matris çarpımı için yeniden şekillendir
	w1 = np.reshape(w1, (observation_space_dim, nn_hidden_unit))
	w2 = np.reshape(w2, (nn_hidden_unit, action_space_dim))

	# Sigmoid fonksiyonu
	sigmoid = lambda x: (1 / (1 + np.exp(-x)))
	# ReLu fonksiyonu
	relu = lambda x: np.maximum(0, x)
	# Softplus fonksiyonu
	softplus = lambda x: np.log1p(np.exp(x))
	# Tanh fonksiyonu
	tanh = lambda x: (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

	# Feed-forward
	h1 = sigmoid(np.dot(state, w1) + b1)

	# Final çıktı
	output = tanh(np.dot(h1, w2) + b2)

	return output[0]

## Hiperparametreler
# Her bireyin gen uzunluğu (ağın parametre sayısı)
n_features = (w1_ndim + w2_ndim + 2)

# Simüle edilecek nesil sayısı
n_generation = 5000

# Popülasyon boyutu (birey sayısı)
n_population = 64

# Seleksiyon turnuvasındaki birey sayısı
selectionTournamentSize = int(n_population//2)

# Çaprazlama ve mutasyon olasılıkları
crossoverProbability = 0.33
individualMutationProbability = 0.25
geneMutationProbability = 0.25

# Uygunluk ve Birey sınıflarını oluştur
creator.create("Fitness", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.Fitness)

# Bireylerin genotiplerini ve popülasyonun basitçe birey listesi olduğunu tanımla
toolbox = base.Toolbox()
toolbox.register("attr_weights_n_biases", random.gauss, 0, 2)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_weights_n_biases, n_features)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Uygunluk fonksiyonunu tanımla
def calculateFitness(individual):
	individualEpisodeRewards = run_env(
		# Aksiyon fonksiyonumuz, bireyin genotipini kullanarak yapay sinir ağını çalıştırır
		act_fn=(lambda state: nn_forward(individual, state)),
		# Her uygunluk hesabında ortamı 1 episode çalıştır
		n_episode=1,
		max_timestep=300
	)
	# tuple tipinde çevirmeli
	return (individualEpisodeRewards, )

# Uygunluk fonksiyonunu kaydet
toolbox.register("evaluate", calculateFitness)

# Bireylerin uygunluk skorunu eşzamanlı ölçebilmek için ThreadPoolExecutor sınıfını toolbox'a register et
# MAX_OPTIMIZER_THREADS = 4
# parallel_executor = concurrent.futures.ThreadPoolExecutor(max_workers=MAX_OPTIMIZER_THREADS)
# toolbox.register("map", parallel_executor.map)

# Hangi çaprazlama, mutasyon ve seçilim yöntemlerinin kullanılacağını tanımla
toolbox.register("mate", tools.cxOnePoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=2, indpb=geneMutationProbability)
toolbox.register("select", tools.selTournament, tournsize=selectionTournamentSize)

# Popülasyonu oluştur ya da yükle
if loadPopulation:
	try:
		initialPopulation = pickle.Unpickler(open(initialPopulationInPath, "rb")).load()
		print("[+] Başlangıç popülasyonu başarıyla yüklendi!")
	except Exception as e:
		print("[x] Başlangıç popülasyonu yüklenemedi: {}".format(e))
		initialPopulation = toolbox.population(n=n_population)
else:
	initialPopulation = toolbox.population(n=n_population)
hallOfFame = tools.HallOfFame(1)

# Fitness istatistiği
stats = tools.Statistics(lambda ind: ind.fitness.values[0])
stats.register("avg", np.mean, axis=0)
stats.register("std", np.std, axis=0)
stats.register("min", np.min, axis=0)
stats.register("max", np.max, axis=0)

# Simülasyon başlasın!
def run_simulation(pop, ngen=1):
	return algorithms.eaSimple(
		population=pop,
		toolbox=toolbox,
		halloffame=hallOfFame,
		stats=stats,
		ngen=ngen,
		cxpb=crossoverProbability,
		mutpb=individualMutationProbability,
		verbose=False
	)

def saveProgress():
	try:
		# Tüm nesillerin popülasyon geçmişini kaydet
		pickle.Pickler(open(allGenerationsFinalPopulationOutPath, "wb")).dump(allGenerationsFinalPopulation)

		# Son popülasyonu kaydet
		pickle.Pickler(open(finalPopulationOutPath, "wb")).dump(population)

		# Simülasyon sonucu ortaya çıkan en iyi bireyi kaydet
		np.save(modelOutPath, np.array(hallOfFame[0]))

		print("[+] Popülasyon geçmişleri, final popülasyon ve bireyi başarıyla kaydedildi!")
	except Exception as e:
		print("[!] Popülasyon geçmişleri, final popülasyon ve bireyi kaydedilemedi {}".format(e))

allGenerationsFinalPopulation = [initialPopulation]
population = initialPopulation
try:
	for currentGeneration in range(n_generation):
		# 1 nesil çalıştır
		population, _ = run_simulation(pop=population, ngen=1)

		# Nesil popülasyonunu listeye ekle
		allGenerationsFinalPopulation.append(population)

		# Uygunluk istatistiklerini al, logla
		record = stats.compile(population)
		print(
			", ".join(
				["Nesil {}".format(currentGeneration+1)] + ["{0} {1:.2f}".format(k, v) for k,v in record.items()]
			)
		)
		if currentGeneration > 0 and ((currentGeneration % int(n_generation//100)) == 0):
			saveProgress()
except KeyboardInterrupt:
	print("[*] Simülasyon yarıda durduruldu..")
	saveProgress()
	exit()

saveProgress()

bestIndividual = hallOfFame[0]
print("[+] En iyi uygunluk değeri: {}".format(bestIndividual.fitness.values[0]))
print("Genotip: ", bestIndividual)

# En iyi bireyle simülasyonu çalıştır
run_env(
	act_fn=(lambda state: nn_forward(bestIndividual, state)),
	n_episode=3,
	render=True
)
