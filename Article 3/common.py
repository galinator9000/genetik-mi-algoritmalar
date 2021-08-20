# Bu Python dosyası projede ortak olarak kullanılan bazı fonksiyon ve sınıfları içerir.
import numpy as np

# Sigmoid aktivasyon fonksiyonu
sigmoid = lambda x: (1 / (1 + np.exp(-x)))

# ReLu aktivasyon fonksiyonu
relu = lambda x: np.maximum(0, x)

# Softplus aktivasyon fonksiyonu
softplus = lambda x: np.log1p(np.exp(x))

# Tanh aktivasyon fonksiyonu
tanh = lambda x: (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

# Yapay sinir ağı sınıfımız
class NeuralNetwork:
	def __init__(self, layer_units):
		self.layer_units = layer_units

	def get_parameter_count(self):
		weight_params_count = sum([
			self.layer_units[i] * self.layer_units[i+1]
			for i in range(0, len(self.layer_units)-1)
		])
		bias_params_count = len(self.layer_units)-1
		return (weight_params_count + bias_params_count)

	# Verilen bireyin genotipini yapay sinir ağı parametreleri olarak kullanarak, state vektörünü ağa feed-forward eder
	def forward(self, individual, state):
		# State'in 2 seviyeli tensör olduğundan emin ol: [?, observation_space_dim]
		if len(state.shape) == 1: state = np.array([state])
		assert len(state.shape) == 2

		# Yapay sinir ağı parametrelerini verilen gen dizisinden çıkar
		geneCurrentIndex = 0
		layer_weights_n_biases = []
		for layer_i in range(0, len(self.layer_units)-1):
			## Katman ağırlıklarını çıkar
			# (Önceki katman nöron sayısı x sonraki katman nöron sayısı)
			weightParamCount = self.layer_units[layer_i] * self.layer_units[layer_i+1]

			# Gen dizisinden çıkar
			layer_weights = np.array(individual[geneCurrentIndex: geneCurrentIndex+weightParamCount])
			geneCurrentIndex += weightParamCount

			## Katman bias değerini çıkar
			layer_bias = individual[geneCurrentIndex]
			geneCurrentIndex += 1

			# Ağırlık değerlerini 1D vektörden, gereken şekle reshape et
			layer_weights = np.reshape(
				layer_weights,
				(
					self.layer_units[layer_i],
					self.layer_units[layer_i+1]
				)
			)
			layer_weights_n_biases.append((layer_weights, layer_bias))

		# Girdi vektörüyle başlayarak, katman çıktılarını sırayla ağırlıklarla çarparak biasları ekle: yani feed-forward!
		hidden_output = state
		for weight, bias in layer_weights_n_biases:
			# Aktivasyon
			hidden_output = sigmoid(
				# x*w + b
				np.dot(hidden_output, weight) + bias
			)

		# Bu noktada hidden_output değerimiz ağın çıktısını taşıyor, argmax ile ilgili aksiyonu alalım.
		return np.argmax(hidden_output)

## Verilen gym ortamını, verilen aksiyon fonksiyonuyla çalıştırmak için ortak bir fonksiyon
def run_gym_environment(env, act_fn, n_episode=1, render=False, n_timestep=None):
	totalRewards = 0

	# Episode döngüsü
	for episode in range(n_episode):
		timestep = 0
		state = env.reset()
		done = False

		def shouldStop():
			# Eğer timestep değeri verilmişse sadece onu dikkate al
			if (n_timestep != None):
				return (timestep >= n_timestep)
			# Verilmemişse ortamın sağladığı tamamlanma değerini kullan
			else:
				return done

		# Timestep döngüsü
		episodeReward = 0
		while not shouldStop():
			if render: env.render()

			# Aksiyon kararı ver
			action = act_fn(state)

			# Ortamda uygula
			state, reward, done, _ = env.step(action)
			episodeReward += reward
			timestep += 1

		if render: print("Episode bitti, ödül {}".format(episodeReward))
		totalRewards += episodeReward

	# Simülasyondan toplanan toplam ödülü çevirir
	return totalRewards
