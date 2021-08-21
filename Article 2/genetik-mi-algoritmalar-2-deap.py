import random
import numpy as np
from deap import algorithms, base, creator, tools

# Tekrarlanabilirlik için seed ayarlanır
SEED_VALUE = 32
random.seed(SEED_VALUE)
np.random.seed(SEED_VALUE)

## Hiperparametreler
# Simüle edilecek nesil sayısı
n_generation = 32

# Popülasyon boyutu (birey sayısı)
n_population = 16

# Her bireyin gen sayısı
n_genes = 16

# Seçilim turnuvasında rastgele seçilecek birey sayısı (k)
selectionTournamentSize = 3

# Birey bazlı mutasyon olasılığı
individualMutationProbability = 0.05

# Gen bazlı mutasyon olasılığı
geneMutationProbability = 0.05

# Çaprazlama olasılığı
crossoverProbability = 0.90

# "Uygunluk" ve "Birey" sınıflarını tanımla, uygunluğun bireye ait bir değer olacağını da tanımla
creator.create("Fitness", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.Fitness)

## Toolbox'ımız
toolbox = base.Toolbox()

# Her bir bireyin her bir genini oluşturacak fonksiyonu toolbox içinde tanımla
# (0/1, boolean)
toolbox.register("initializeGene", random.randint, 0, 1)

# Bireyleri oluşturacak fonksiyonu toolbox içinde tanımla
toolbox.register("initializeIndividual", tools.initRepeat, creator.Individual, toolbox.initializeGene, n=n_genes)

# Popülasyonu oluşturacak fonksiyonu toolbox içinde tanımla
toolbox.register("initializePopulation", tools.initRepeat, list, toolbox.initializeIndividual, n=n_population)

# Gen, birey ve popülasyon oluşturan toolbox fonksiyonlarımızı test edelim
print(
	"+ Örnek gen",
	toolbox.initializeGene()
)
print(
	"\n+ Örnek birey\n",
	np.array(toolbox.initializeIndividual())
)
print(
	"\n+ Örnek popülasyon (n=5)\n",
	np.array(toolbox.initializePopulation(n=5))
)

# Uygunluk fonksiyonunu toolbox içinde tanımla (gen değerlerinin toplamı)
toolbox.register(
	"evaluate",
	lambda individual_genes: (sum(individual_genes), )
)

# Çaprazlama, mutasyon ve seçilimi uygulayacak fonksiyonlarımızı toolbox yoluyla tanımlayalım
# Çaprazlama metodu: Tek noktalı çaprazlama
toolbox.register("mate", tools.cxOnePoint)

# Mutasyon metodu: Bit flip mutasyonu
toolbox.register("mutate", tools.mutFlipBit, indpb=geneMutationProbability)

# Seçilim metodu: Turnuva seçilimi, k = selectionTournamentSize
toolbox.register("select", tools.selTournament, tournsize=selectionTournamentSize)

# Uygunluğu ölçen ve çaprazlama, mutasyon, seçilim gibi genetik operatörleri uygulayacak toolbox fonksiyonlarımızı test edelim
# Test için iki birey tanımlayalım
ind1 = toolbox.initializeIndividual()
ind2 = toolbox.initializeIndividual()

print(
	"\n+ Uygunluk skoru\n",
	np.array(ind1), toolbox.evaluate(ind1),
	"\n",
	np.array(ind2), toolbox.evaluate(ind2)
)

print(
	"\n+ Tek Noktalı Çaprazlama\n",
	np.array(ind1), "Birey 1\n",
	np.array(ind2), "Birey 2\n",
	"\nYeni bireyler\n",
	np.array(toolbox.mate(ind1, ind2))
)

print(
	"\n+ Bit Flip Mutasyonu\n",
	np.array(ind1), "Birey\n",
	np.array(toolbox.mutate(ind1)[0]), "Mutasyon uygulanmış hali",
)

print("\n+ Turnuva Seçilimi (parent selection) (k=2)")
print("\nPopülasyon:")

# Örnek popülasyon tanımla
examplePopulation = toolbox.initializePopulation(n=6)

# Bireylerin uygunluklarını hesaplayıp üstlerine yaz (seçilim operatörü bu değeri okuyarak seçiyor)
fitnesses = [toolbox.evaluate(ind) for ind in examplePopulation]
for ind, fitnessScore in zip(examplePopulation, fitnesses):
	ind.fitness.values = fitnessScore

# Popülasyonu uygunluk skorlarıyla yazdır
for ind in examplePopulation:
	print(np.array(ind), ind.fitness.values[0])

# Seçilim operatörünü uygula
# tournsize, daha önce gösterdiğimiz k parametresi, yani uygunluğa bakılmadan kaç birey seçilip aralarında turnuva yapılacağını belirler
# Burada geçen k parametresi ise turnuvanın kaç defa gerçekleşeceği
selectedInd = toolbox.select(examplePopulation, k=1, tournsize=2)[0]
print("\nSeçilen birey: ", selectedInd, toolbox.evaluate(selectedInd)[0], "\n")

## Simülasyonu çalıştırma zamanı!
print("--- Simülasyon zamanı! ---")

# Başlangıç popülasyonunu oluştur
initialPopulation = toolbox.initializePopulation(n=n_population)

# Simülasyon boyunca en uygun bireyin kaydını tutacak objemiz
hallOfFame = tools.HallOfFame(1)

# Uygunluk istatistiklerini tutacak objemiz
stats = tools.Statistics(lambda ind: ind.fitness.values[0])
stats.register("min", np.min)
stats.register("max", np.max)

print("+ Başlangıç popülasyonu:")
print(np.array(initialPopulation))

# Simülasyon başlasın!
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

bestIndividual = hallOfFame[0]
print("\n+ En iyi uygunluk sağlayan bireyin skoru: {}".format(bestIndividual.fitness.values[0]))
print("Genotipi: ", bestIndividual)

print("+ Final popülasyon:")
print(np.array(finalPopulation))