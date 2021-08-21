import random, math, os
import matplotlib.pyplot as plt
import numpy as np
from deap import algorithms, base, creator, tools

# Tekrarlanabilirlik için seed ayarlanır
SEED_VALUE = 21
random.seed(SEED_VALUE)
np.random.seed(SEED_VALUE)

## Travelling Salesman Problem
class TSP:
	# Problemi tanımla
	def __init__(self, n_city=16, WIDTH=640, HEIGHT=640, save_figures_path=None):
		if n_city%2 != 0:
			n_city += 1
			print("n_city değeri çift yapıldı: {}".format(n_city))
		self.n_city = n_city

		# Şehirlerin konumlarını barındıracak matrisi oluştur
		self.city_xys = np.concatenate(
			[
				# X koordinatları
				np.random.randint(0, WIDTH, (self.n_city, 1)),
				# Y koordinatları
				np.random.randint(0, HEIGHT, (self.n_city, 1))
			],
			axis=1
		)

		# Figürler kaydedilecekse klasörü oluştur
		self.save_figures_path = save_figures_path
		if save_figures_path != None:
			if not os.path.isdir(save_figures_path):
				os.mkdir(save_figures_path)

	# Problem figür penceresini göster
	def show(self):
		self.fig = plt.gcf()
		self.fig.show()

	# Problemi ve rotayı (verilirse) çiz
	def draw(self, routePath=None, generation=0):
		plt.clf()

		# Şehirleri saçılım grafiğiyle 2D ortamda göster
		plt.scatter(
			self.city_xys[:, 0],
			self.city_xys[:, 1]
		)

		# Rota verilmişse çiz
		if routePath != None:
			# Her şehirden geçildiğine emin oL
			assert (len(routePath) == self.n_city) and (sorted(list(routePath)) == sorted(list(range(self.n_city))))
			for c in range(len(routePath)-1):
				c1x, c1y = self.city_xys[routePath[c]]
				c2x, c2y = self.city_xys[routePath[c+1]]
				plt.plot(
					(c1x, c2x),
					(c1y, c2y)
				)

		# Canvas'ı güncelle
		self.fig.canvas.draw()

		# Konum değeri verilmişse figürü kaydet
		if self.save_figures_path != None:
			plt.savefig(os.path.join(self.save_figures_path, "{}.png".format(generation)))

	# Verilen rotanın toplam mesafesini hesapla
	def pathDistance(self, routePath):
		totalDistance = 0
		for c in range(len(routePath)-1):
			# Şehir koordinatlarını çıkar
			[c1x, c1y], [c2x, c2y] = self.city_xys[routePath[c:c+2]]

			# Uzaklığı hesapla
			distance = math.sqrt(
				((c2x - c1x) ** 2)
				+ ((c2y - c1y) ** 2)
			)
			totalDistance += distance
		return totalDistance

# Problem ortamını oluşturalım
tsp = TSP(
	n_city=20,
	# save_figures_path="tsp_stages/"
)

## Hiperparametreler
# Simüle edilecek nesil sayısı
n_generation = 600
# Simülasyon aşama sayısı
n_generation_stage = 20
# Aşama başı nesil sayısı
n_generation_per_stage = (n_generation // n_generation_stage)

# Popülasyon boyutu (birey sayısı)
n_population = 32

# Her bireyin gen sayısı (şehir sayısı)
n_genes = tsp.n_city

# Seçilim turnuvasında rastgele seçilecek birey sayısı (k)
selectionTournamentSize = 7

# Birey bazlı mutasyon olasılığı
individualMutationProbability = 0.15

# Gen bazlı mutasyon olasılığı
geneMutationProbability = 0.15

# Çaprazlama olasılığı
crossoverProbability = 0.80

# "Uygunluk" ve "Birey" sınıflarını tanımla, uygunluğun bireye ait bir değer olacağını da tanımla
creator.create("Fitness", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.Fitness)

## Toolbox'ımız
toolbox = base.Toolbox()

# Bireyleri oluşturacak fonksiyon
def initializeIndividual():
	# Şehir indekslerini üret
	route = list(range(0, n_genes))

	# Karıştır!
	random.shuffle(route)

	# Bireyi oluşturup genleri ata
	individual = creator.Individual()
	individual.extend(route)
	return individual

# Popülasyonu oluşturacak fonksiyonu toolbox içinde tanımla
toolbox.register("initializePopulation", tools.initRepeat, list, initializeIndividual, n=n_population)

# Uygunluk fonksiyonunu toolbox içinde tanımla (toplam rota mesafesinin negatif değeri)
toolbox.register(
	"evaluate",
	lambda individual_genes: (-tsp.pathDistance(individual_genes), )
)

# Çaprazlama, mutasyon ve seçilimi uygulayacak fonksiyonlarımızı toolbox yoluyla tanımlayalım
# Çaprazlama metodu: Ordered crossover (OX)
toolbox.register("mate", tools.cxOrdered)

# Mutasyon metodu: Shuffle mutation
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=geneMutationProbability)

# Seçilim metodu: Turnuva seçilimi, k = selectionTournamentSize
toolbox.register("select", tools.selTournament, tournsize=selectionTournamentSize)

## Simülasyonu çalıştırma zamanı!
print("--- Simülasyon zamanı! ---")

# Başlangıç popülasyonunu oluştur
population = toolbox.initializePopulation(n=n_population)

# Simülasyon boyunca en uygun bireyin kaydını tutacak objemiz
hallOfFame = tools.HallOfFame(1)

# Uygunluk istatistiklerini tutacak objemiz
stats = tools.Statistics(lambda ind: ind.fitness.values[0])
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)

print("+ Başlangıç popülasyonu:")
print(np.array(population))

# Simülasyonu adım adım işletmek için fonksiyonumuz
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

# Problem ortam penceresini göster
tsp.show()

# Bireylerin uygunluklarını hesaplayıp üstlerine yaz (ilk aşamada gerekli)
for ind, fitnessScore in zip(population, [toolbox.evaluate(ind) for ind in population]):
	ind.fitness.values = fitnessScore
hallOfFame.update(population)

# Simülasyon başlasın!
for stage in range(0, n_generation_stage):
	generation = stage * n_generation_per_stage

	# En iyi rotayı logla
	print("\n--- Nesil {} ---".format(generation))
	print("+ En iyi uygunluk sağlayan bireyin skoru: {}".format(hallOfFame[0].fitness.values[0]))
	print("Genotipi: ", hallOfFame[0])

	# En iyi rotayı çiz
	tsp.draw(
		routePath=hallOfFame[0],
		generation=generation
	)

	# Simülasyonu çalıştır, popülasyonu güncelle
	population, logs = run_simulation(
		pop=population,
		ngen=n_generation_per_stage
	)

finalPopulation = population

print("+ Final popülasyon:")
print(np.array(finalPopulation))

# En iyi rotayı çiz
bestIndividual = hallOfFame[0]
tsp.draw(
	routePath=bestIndividual,
	generation=n_generation
)
