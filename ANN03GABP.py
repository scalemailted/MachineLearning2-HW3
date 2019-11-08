import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
import copy
import random
from sklearn.metrics import accuracy_score
#Here solving cancer{Benign, Malignant} toy problem using ANN {Back prop based}


class NeuralNet:
    def __init__(self, config):
        self.layers = config['layers'];
        self.alpha = config['alpha'];
        self.target_mse = config['target_mse'];
        self.max_epoch = config['max_epoch'];
        self.gamma = config['gamma'];
        self.error_log = [];
        self.epoch_log = [];
        self.test_log = [];
        self.test_error_log = [];
        self.test_accuracy = [];
        self.train_accuracy = [];
        self.beta_error_log = [];
    #
    def setup_training(self):
        self.min_error = float('inf');
        self.min_error_epoch = -1;
        self.epoch = 0;
        self.mse = float('inf');
        self.init_betas();
        self.init_transmitters();
        self.init_Zs();
        self.init_delta_errors();
        self.init_momentum();
    #
    def setup_testing(self):
        self.init_transmitters();
        self.init_Zs();
    #
    def set_data(self, X, Y):
        self.X = X;
        self.Y = Y.T;
        self.Nx, self.P = X.shape;
        self.Ny, self.K = Y.shape;
    #
    def init_betas(self):
        self.B = np.empty( len(self.layers)-1, dtype=object )              
        for i in range( len(self.B) ):
            self.B[i] = 1.4 * np.random.rand( self.layers[i]+1, self.layers[i+1] ) - 0.7; 
    #
    def init_transmitters(self):
        self.T = np.empty(len(self.layers), dtype=object)              
        for i in range(len(self.layers)):
            self.T[i] = np.ones( self.layers[i]  );
    #
    def init_Zs(self):
        self.Z = np.empty( len(self.layers), dtype=object)              
        for i in range( len(self.layers) ):
            self.Z[i] = np.zeros( self.layers[i]+1  );  
        self.Z[0] = (np.append(self.X, np.ones([self.Nx,1]), axis=1)).T
    #
    def init_delta_errors(self):
        self.d = np.empty( len(self.layers), dtype=object) 
        for i in range( len(self.layers) ):
            self.d[i] = np.zeros( self.layers[i]  ); 
    #
    def init_momentum(self):
        self.velocity = np.empty(len(self.layers)-1, dtype=object)              
        for i in range(len(self.velocity)):
            self.velocity[i] = np.zeros( [self.layers[i]+1,self.layers[i+1]] );
    #
    def train(self, X, Y):
        self.set_data(X,Y);
        self.setup_training();
        error_data = []
        epoch_data = []
        beta_data = [];
        accuracy = []
        config_GA['data'] = (X,Y,self)
        config_GA['layers'] = self.layers;  
        ga = GeneticAlgorithm(config_GA);
        ga.execute()
        error_data = []
        epoch_data = []
        beta_data = [];
        accuracy = []
        while self.mse > self.target_mse and self.epoch < self.max_epoch:
            CSqErr = 0;
            self.forward_propagation();
            CSqErr += np.sum( np.sum( (self.Y - self.Z[-1])**2, 0) ) 
            CSqErr = CSqErr / self.layers[-1]
            accuracy.append( self.get_accuracy() );
            #print(CSqErr)
            self.get_delta();
            self.update_weights();
            CSqErr = CSqErr / self.Nx;
            self.mse = CSqErr;
            self.epoch += 1;
            #print('mse: '+str(self.mse)+', epoch: '+ str(self.epoch));
            if self.mse < self.min_error:
                self.min_error = self.mse;
                self.min_error_epoch = self.epoch;
            error_data.append(self.mse);
            epoch_data.append(self.epoch);
            beta_data.append( copy.deepcopy(self.B));
        self.error_log.append(error_data);
        self.epoch_log.append(epoch_data);
        self.betas = beta_data;
        self.train_accuracy.append(accuracy);
        #plt.plot(self.epo,self.err);
        #plt.show()
    #
    def forward_propagation(self):
        for i in range( len(self.layers)-1 ):
            self.T[i+1] = self.B[i].T @ self.Z[i];
            if (i+1)<len(self.layers)-1:
                t = 1/(1+np.exp(-self.T[i+1]))
                ones = np.ones(self.Nx)
                self.Z[i+1] = (np.column_stack([t.T,ones])).T
            else:
                self.Z[i+1] = (1/(1+np.exp(-self.T[i+1])))
    #
    def get_delta(self):
        self.d[-1] = (self.Z[-1]-self.Y) * self.Z[-1] * (1 - self.Z[-1])     #delta term error from output
        for i in reversed( range(1, len(self.layers)-1) ):
            W = self.Z[i][:-1] * (1 - self.Z[i][:-1]);
            D = self.d[i+1];
            self.d[i] = np.ones([self.Nx,self.layers[i]])
            for m in range(self.Nx):
                #print('iteration ' + str(i)+'iteration: '+str(m) )
                self.d[i][m] = (W.T[m] * np.sum( (D.T[m]  * self.B[i][:-1] ), 1  ));
            self.d[i] = self.d[i].T #TODO This is transposed wrong, find out why
    #
    def update_weights(self):
        for i in range( len(self.layers)-1 ):
            W = self.Z[i][:-1].T;
            V1 = np.zeros( [self.layers[i],self.layers[i+1]] );
            V2 = np.zeros( [1,self.layers[i+1]] );
            D = self.d[i+1].T;
            for m in range(self.Nx):
                #print("i: "+str(i) + ", m: "+str(m))
                V1 = V1 + (W[m].reshape(-1,1) @ D[m].reshape(1,-1));
                V2 = V2 + D[m]
            self.velocity[i][:-1]  = self.gamma * self.velocity[i][:-1] - (self.alpha/self.Nx) * V1; 
            self.velocity[i][ -1]  = self.gamma * self.velocity[i][ -1] - (self.alpha/self.Nx) * V2;  
            self.B[i][:-1] = self.B[i][:-1] + self.velocity[i][:-1]; 
            self.B[i][ -1] = self.B[i][ -1] + self.velocity[i][ -1];
    #
    def test(self, X, Y):
        self.set_data(X,Y);
        #self.setup_training()
        self.setup_testing();
        CSqErr = 0;
        self.forward_propagation()
        CSqErr += np.sum( np.sum( (self.Y - self.Z[-1])**2, 0) ) 
        CSqErr = CSqErr / self.layers[-1]
        CSqErr = CSqErr / self.Nx;
        #print("Testing error: " + str(CSqErr))
        self.test_log.append(CSqErr);
        #Forward propagate
        #print("prediction: " + str(self.Z[-1]))
        #return CSqErr
    #
    def testB(self, X, Y):
        beta_errs = []
        accuracy = []
        self.set_data(X,Y);
        for b in self.betas:
            CSqErr = 0;
            self.B = b;
            self.setup_testing();
            self.forward_propagation();
            CSqErr += np.sum( np.sum( (self.Y - self.Z[-1])**2, 0) ) 
            CSqErr = CSqErr / self.layers[-1]
            CSqErr = CSqErr / self.Nx;
            beta_errs.append(CSqErr)
            accuracy.append( self.get_accuracy() );
        self.beta_error_log.append(beta_errs);
        self.test_accuracy.append( accuracy )  
    #
    def get_accuracy(self):
        y = self.Y.T
        y_pred = np.zeros_like(self.Z[-1].T)
        y_pred[np.arange(len(self.Z[-1].T)), self.Z[-1].T.argmax(1)] = 1
        score = accuracy_score(y, y_pred)
        return score


config_GA = {
    'target_value':0.005, 
    'max_generations':1000, 
    'layers':(4,4,4,4,3)
    #'data': (X, Y, NeuralNet(config))
}

class GeneticAlgorithm:
    def __init__(self, config):
        self.population_size = 200;
        self.elite_rate = .10;
        self.cross_over = .80;
        self.mutation   = .05; 
        self.population = [];
        self.target_value = config['target_value'];
        self.max_generations = config['max_generations'];
        self.layers = config['layers']; 
        self.data = config['data'];
    #
    def initialize_population(self ):
        #print('initialize population');
        for i in range( self.population_size ):
            self.population.append( Betas( self.layers) );
    #   
    def calculate_fitness(self):
        #print('compute fitness');
        for b in self.population:
            x,y,ann = self.data;
            b.get_mse(x,y,ann);
        self.population.sort( key=lambda b: b.score  )
        
    #
    def getElite(self):
        #print('get elite');
        ielite = int(self.population_size * self.elite_rate)
        self.pop_elite = copy.deepcopy( self.population[0:ielite]); 
    #
    def getNonelite(self):
        #print('get nonelites');
        nonelite_size = self.population_size - int(self.population_size * self.elite_rate);
        crossover_size = int(nonelite_size * self.cross_over);
        self.pop_nonelite = [];
        for i in range(0,crossover_size,2):
            self.getCrossovers();
        while (len(self.pop_nonelite) < nonelite_size ):
            self.getRandoms();
    #
    def getCrossovers(self):
        #print('get crossovers')
        b1 = self.selectOne();
        b2 = self.selectOne();
        index = np.random.randint( len(self.layers)-1 );
        b1.swap(index,b2);
        self.pop_nonelite.append(b1);
        self.pop_nonelite.append(b2);
    #
    def getRandoms(self):
        #print('get randoms');
        b = Betas(self.layers);
        self.pop_nonelite.append(b)
    #
    def mutatePopulation(self):
        #print('mutate population');
        mutation_size = int( len(self.pop_nonelite) * self.mutation)
        for i in range(mutation_size):
            index = np.random.randint( len(self.pop_nonelite) );
            self.pop_nonelite[index].mutate();
    #
    def selectOne(self):
        max = sum([b.score for b in self.population])
        selection_probs = [b.score/max for b in self.population][::-1]
        return self.population[np.random.choice(len(self.population), p=selection_probs)]
    #
    def execute(self):
        self.initialize_population();
        self.calculate_fitness();
        counter = 0;
        while self.population[0].score < self.target_value or counter < self.max_generations:
            print("Generation: " + str(counter) + ", top score: " + str(self.population[0].score))
            self.getElite();
            self.getNonelite();
            self.population = self.pop_elite + self.pop_nonelite;
            self.calculate_fitness();
            counter += 1;
    
    #def test(self, testX, testY, ann):
    #    b = copy.deepcopy( self.population[0]); 
    #    b.get_mse(testX,testY,ann);
    #    return b.score;
        

        


class Betas:
    def __init__(self, layers):
        self.layers = np.empty( len(layers)-1, dtype=object ) 
        self.score = float('inf');
        self.accuracy = 0;             
        for i in range( len(self.layers) ):
            self.layers[i] = 4 * np.random.rand( layers[i]+1,layers[i+1] ) - 2.0; 
    #
    #def get_mse(self,X,Y,Z,T):
    #    Nx,_ = X.shape;
    #    CSqErr = 0;
    #    for j in range(Nx):
    #        Z[0] = (np.append(X[j],1)).T
    #        for i in range( len( self.layers )-1 ):
    #            T[i+1] = B[i].T @ Z[i];
    #            if (i+1)<len(self.layers)-1:
    #                t = 1/(1+np.exp(-T[i+1]))
    #                Z[i+1] = np.append(t,1) 
    #            else:
    #                Z[i+1] = (1/(1+np.exp(-T[i+1])))
    #    CSqErr += np.sum( np.sum( (Y.T - Z[-1])**2, 0) ) 
    #    CSqErr = CSqErr / self.layers[-1]
    #    CSqErr = CSqErr / Nx;
    #    return CSqErr 
    def setup_mse(self, X,Y, ann):
        ann.set_data(X,Y)
        ann.setup_training()
    #
    def get_mse(self, X,Y, ann):
        #ann.set_data(X,Y)
        #ann.setup_training()
        ann.B = self.layers;
        ann.test(X,Y);
        self.score = ann.test_log[-1];#.pop();
        self.accuracy = ann.get_accuracy();
    #
    def swap(self, index, otherBetas):
        temp = self.layers[index];
        self.layers[index] = otherBetas.layers[index];
        otherBetas.layers[index] = temp;
    #
    def mutate(self):
        ilayer = np.random.randint( len(self.layers) );
        rows, cols = self.layers[ilayer].shape;
        i = np.random.randint(rows);
        j = np.random.randint(cols);
        self.layers[ilayer][i][j] = 4 * np.random.random() - 2.0;

        
def create_layers(inputs, hiddens, outputs):
    layers = [];
    layers.append(inputs);
    for i in range(hiddens):
        nodes = random.randint(2,20);
        layers.append(nodes);
    layers.append(outputs);
    return layers


#Test 
config = {
    'layers': create_layers(inputs=4,hiddens=3,outputs=3),
    'alpha': 0.2,
    'target_mse': 0.0001,
    'max_epoch': 2000,
    'gamma': 0.8
}

def main():
    labels = np.loadtxt('iris.data.txt', delimiter=',', dtype=np.str, usecols=[4])
    Y = OneHotEncoder(sparse=False).fit_transform( labels.reshape(-1,1) )
    X = np.loadtxt('iris.data.txt', delimiter=',', usecols=[0,1,2,3])
    folds = KFold(n_splits=10, shuffle=True)
    folds.get_n_splits(X)
    ann = NeuralNet(config);
    counter = 0;
    for train_index, test_index in folds.split(X):
        #print("TRAIN:", train_index, "TEST:", test_index)
        counter +=1; 
        print("Fold: " + str(counter));
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        ann.train(X_train, Y_train);
        ann.test(X_test, Y_test);
        ann.testB(X_test, Y_test);
        print(len(ann.error_log))
    #print(ann.error_log)
    #print("Test Error (Mean):" + str( np.mean(ann.error_log,axis=0)[-1] ))
    print("Test Error (Mean):" + str(np.mean( [err[-1] for err in ann.error_log] )))
    train_err = np.mean(np.array(ann.error_log).T,axis=1)
    test_err = np.mean(np.array(ann.beta_error_log).T,axis=1)
    epo = ann.epoch_log[0]
    fig = plt.figure()
    fig.suptitle('Neural Net/GA+BP [3 Hidden Layers]');
    plt.plot(epo, train_err, label="Train Error");
    plt.plot(epo, test_err, label='Test Error');
    plt.xlabel('Generations')
    plt.ylabel('Avg Error (10CV)')
    plt.legend()
    plt.savefig('./output/ANN03GABP-Error.png')
    #accuracy
    train_acc = np.mean(np.array(ann.train_accuracy).T,axis=1)
    test_acc = np.mean(np.array(ann.test_accuracy).T,axis=1)
    epo = ann.epoch_log[0]
    fig = plt.figure()
    fig.suptitle('Neural Net/GA+BP [3 Hidden Layers]');
    plt.plot(epo, train_acc, label="Train Accuracy");
    plt.plot(epo, test_acc, label='Test Accuracy');
    plt.xlabel('Epochs')
    plt.ylabel('Avg Accraucy (10CV)')
    plt.legend()
    plt.savefig('./output/ANN03GABP-Accuracy.png')
    

if __name__ == "__main__":
    # execute only if run as a script
    main()

        



