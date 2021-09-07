import random, gym
import numpy as np
from deap import algorithms, base, creator, tools

trainingSession = 1
modelInPath = "models/{}/bipedalWalkerBest".format(trainingSession)

# Kaydedilen modeli yükle
bestIndividual = np.load(modelInPath if modelInPath.endswith(".npy") else (modelInPath+".npy"))

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

# En iyi bireyle simülasyonu çalıştır
run_env(
	act_fn=(lambda state: nn_forward(bestIndividual, state)),
	n_episode=10,
	render=True
)
env.close()
