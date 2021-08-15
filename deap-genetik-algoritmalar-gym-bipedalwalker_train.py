import random, gym
import numpy as np
from deap import algorithms, base, creator, tools

modelOutPath = "models/bipedalWalkerBest"

## Simülasyon ortamı
env = gym.make("BipedalWalker-v3")
observation_space_dim = env.observation_space.shape[0]
action_space_dim = env.action_space.shape[0]

# Tekrarlanabilirlik için seed ayarlanır
SEED_VALUE = 64
random.seed(SEED_VALUE)
np.random.seed(SEED_VALUE)
env.seed(SEED_VALUE)

# Simülasyonu çalıştırmak için ortak fonksiyon
def run_env(act_fn, n_episode=1, render=False):
	# Episode döngüsü
	totalRewards = 0
	for episode in range(n_episode):
		state = env.reset()
		done = False

		# Timestep döngüsü
		while not done:
			if render: env.render()

			# act_fn adlı fonksiyon parametreyi çağırarak aksiyonu al
			action = act_fn(state)
			next_state, reward, done, info = env.step(action)

			totalRewards += reward
			state = next_state
	return totalRewards

## Yapay sinir ağı
# Ara katman ünite sayısı
nn_hidden_unit = 8
w1_ndim = (observation_space_dim*nn_hidden_unit)
w2_ndim = (nn_hidden_unit*action_space_dim)
w3_ndim = (nn_hidden_unit*action_space_dim)

# Verilen bireyin genotipini yapay sinir ağı parametreleri olarak kullanarak, state vektörünü feed-forward eder
def nn_forward(individual, state):
	# State'in 2 seviyeli array olduğundan emin ol: [?, observation_space_dim]
	if len(state.shape) == 1: state = np.array([state])
	assert len(state.shape) == 2

	# Yapay sinir ağı parametrelerini genotip olan 1D vektörden çıkar
	arr = np.array(individual)
	w1 = arr[:w1_ndim]
	b1 = arr[w1_ndim]
	w_mu = arr[w1_ndim+1 : w1_ndim+1+w2_ndim]
	b_mu = arr[w1_ndim+1+w2_ndim]
	w_sigma = arr[w1_ndim+1+w2_ndim+1:-1]
	b_sigma = arr[-1]

	# Ağırlıkları matris çarpımı için yeniden şekillendir
	w1 = np.reshape(w1, (observation_space_dim, nn_hidden_unit))
	w_mu = np.reshape(w_mu, (nn_hidden_unit, action_space_dim))
	w_sigma = np.reshape(w_sigma, (nn_hidden_unit, action_space_dim))

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

	# Normal dağılım için Mu & Sigma çıktıları
	mu_output = tanh(np.dot(h1, w_mu) + b_mu)
	sigma_output = relu(np.dot(h1, w_sigma) + b_sigma)

	# Normal dağılımdan sample alarak aksiyonu döndür
	output = np.random.normal(loc=mu_output[0], scale=sigma_output[0])
	return output

## Hiperparametreler
# Her bireyin gen uzunluğu (ağın parametre sayısı)
n_features = (w1_ndim + w2_ndim + w3_ndim + 3)

# Simüle edilecek nesil sayısı
n_generation = 1

# Popülasyon boyutu (birey sayısı)
n_population = 32

# Seleksiyon turnuvasındaki birey sayısı
selectionTournamentSize = 3

# Çaprazlama ve mutasyon olasılıkları
crossingoverProbability = 0.50
mutationProbability = 0.10

# Uygunluk ve Birey sınıflarını oluştur
creator.create("Fitness", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.Fitness)

# Bireylerin genotiplerini ve popülasyonun basitçe birey listesi olduğunu tanımla
toolbox = base.Toolbox()
toolbox.register("attr_weights_n_biases", random.gauss, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_weights_n_biases, n_features)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Uygunluk fonksiyonunu tanımla
def calculateFitness(individual):
	individualEpisodeRewards = run_env(
		# Aksiyon fonksiyonumuz, bireyin genotipini kullanarak yapay sinir ağını çalıştırır
		act_fn=(lambda state: nn_forward(individual, state)),
		# Her uygunluk hesabında ortamı 1 episode çalıştır
		n_episode=1
	)
	# tuple tipinde çevirmeli
	return (individualEpisodeRewards, )

# Uygunluk fonksiyonunu kaydet
toolbox.register("evaluate", calculateFitness)

# Hangi çaprazlama, mutasyon ve seçilim yöntemlerinin kullanılacağını tanımla
toolbox.register("mate", tools.cxUniform, indpb=0.50)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=2, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=selectionTournamentSize)

# Popülasyonu oluştur
population = toolbox.population(n=n_population)
stats = tools.Statistics(lambda ind: ind.fitness.values[0])
hallOfFame = tools.HallOfFame(1)

## GA simülasyonu öncesi rastgele bir bireyle ortamı çalıştır
run_env(
	act_fn=(lambda state: nn_forward(
		random.choice(population),
		state
	)),
	n_episode=5,
	render=True
)

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

best = hallOfFame[0]
print("[+] En iyi uygunluk değeri: {}".format(best.fitness.values[0]))
print("Genotip: ", best)

# Simülasyon sonucu ortaya çıkan en iyi bireyi kaydet
try:
	np.save(modelOutPath, np.array(best))
	print("[+] Model başarıyla kaydedildi!")
except Exception as e:
	print("[!] Model kaydedilemedi: {}".format(e))

# En iyi bireyle simülasyonu çalıştır
run_env(
	act_fn=(lambda state: nn_forward(best, state)),
	n_episode=10,
	render=True
)
env.close()
