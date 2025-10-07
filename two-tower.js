class BasicTwoTowerModel {
    constructor(numUsers, numItems, embeddingDim) {
        this.numUsers = numUsers;
        this.numItems = numItems;
        this.embeddingDim = embeddingDim;
        
        // Initialize embedding tables with small random values
        // Basic architecture: simple embedding lookup for both towers
        this.userEmbeddings = tf.variable(
            tf.randomNormal([numUsers, embeddingDim], 0, 0.05), 
            true, 
            'basic_user_embeddings'
        );
        
        this.itemEmbeddings = tf.variable(
            tf.randomNormal([numItems, embeddingDim], 0, 0.05), 
            true, 
            'basic_item_embeddings'
        );
        
        // Adam optimizer for stable training
        this.optimizer = tf.train.adam(0.001);
    }
    
    // User tower: simple embedding lookup
    userForward(userIndices) {
        return tf.gather(this.userEmbeddings, userIndices);
    }
    
    // Item tower: simple embedding lookup  
    itemForward(itemIndices) {
        return tf.gather(this.itemEmbeddings, itemIndices);
    }
    
    // Scoring function: dot product between user and item embeddings
    score(userEmbeddings, itemEmbeddings) {
        return tf.sum(tf.mul(userEmbeddings, itemEmbeddings), -1);
    }
    
    async trainStep(userIndices, itemIndices) {
        return await tf.tidy(() => {
            const userTensor = tf.tensor1d(userIndices, 'int32');
            const itemTensor = tf.tensor1d(itemIndices, 'int32');
            
            // In-batch sampled softmax loss
            const loss = () => {
                const userEmbs = this.userForward(userTensor);
                const itemEmbs = this.itemForward(itemTensor);
                
                // Compute similarity matrix: batch_size x batch_size
                const logits = tf.matMul(userEmbs, itemEmbs, false, true);
                
                // Labels: diagonal elements are positives
                const labels = tf.oneHot(
                    tf.range(0, userIndices.length, 1, 'int32'), 
                    userIndices.length
                );
                
                // Softmax cross entropy loss
                const loss = tf.losses.softmaxCrossEntropy(labels, logits);
                return loss;
            };
            
            // Compute gradients and update embeddings
            const { value, grads } = this.optimizer.computeGradients(loss);
            
            this.optimizer.applyGradients(grads);
            
            return value.dataSync()[0];
        });
    }
    
    getUserEmbedding(userIndex) {
        return tf.tidy(() => {
            return this.userForward([userIndex]).squeeze();
        });
    }
    
    async getScoresForAllItems(userEmbedding) {
        return await tf.tidy(() => {
            // Compute dot product with all item embeddings
            const scores = tf.dot(this.itemEmbeddings, userEmbedding);
            return scores.dataSync();
        });
    }
    
    getItemEmbeddings() {
        return this.itemEmbeddings;
    }
}

class DeepTwoTowerModel {
    constructor(numUsers, numItems, embeddingDim, numGenres) {
        this.numUsers = numUsers;
        this.numItems = numItems;
        this.embeddingDim = embeddingDim;
        this.numGenres = numGenres;
        
        // Initialize embedding tables
        this.userEmbeddings = tf.variable(
            tf.randomNormal([numUsers, embeddingDim], 0, 0.05), 
            true, 
            'deep_user_embeddings'
        );
        
        this.itemEmbeddings = tf.variable(
            tf.randomNormal([numItems, embeddingDim], 0, 0.05), 
            true, 
            'deep_item_embeddings'
        );
        
        // User tower MLP layers
        this.userHiddenLayer = tf.layers.dense({
            units: embeddingDim * 2,
            activation: 'relu',
            useBias: true,
            kernelInitializer: 'glorotNormal'
        });
        
        this.userOutputLayer = tf.layers.dense({
            units: embeddingDim,
            activation: 'linear',
            useBias: true,
            kernelInitializer: 'glorotNormal'
        });
        
        // Item tower MLP layers (with genre features)
        this.itemHiddenLayer = tf.layers.dense({
            units: embeddingDim * 2,
            activation: 'relu',
            useBias: true,
            kernelInitializer: 'glorotNormal'
        });
        
        this.itemOutputLayer = tf.layers.dense({
            units: embeddingDim,
            activation: 'linear',
            useBias: true,
            kernelInitializer: 'glorotNormal'
        });
        
        // Adam optimizer
        this.optimizer = tf.train.adam(0.001);
    }
    
    // Deep user tower: embedding → hidden layer → output
    userForward(userIndices) {
        return tf.tidy(() => {
            const userEmbs = tf.gather(this.userEmbeddings, userIndices);
            const hidden = this.userHiddenLayer.apply(userEmbs);
            const output = this.userOutputLayer.apply(hidden);
            return output;
        });
    }
    
    // Deep item tower: embedding + genre features → hidden layer → output
    itemForward(itemIndices, genreFeatures) {
        return tf.tidy(() => {
            const itemEmbs = tf.gather(this.itemEmbeddings, itemIndices);
            const genreTensor = tf.tensor2d(genreFeatures, [itemIndices.length, this.numGenres]);
            
            // Concatenate item embeddings with genre features
            const combined = tf.concat([itemEmbs, genreTensor], 1);
            
            const hidden = this.itemHiddenLayer.apply(combined);
            const output = this.itemOutputLayer.apply(hidden);
            return output;
        });
    }
    
    // Scoring function: dot product between user and item embeddings
    score(userEmbeddings, itemEmbeddings) {
        return tf.sum(tf.mul(userEmbeddings, itemEmbeddings), -1);
    }
    
    async trainStep(userIndices, itemIndices, genreFeatures) {
        return await tf.tidy(() => {
            const userTensor = tf.tensor1d(userIndices, 'int32');
            const itemTensor = tf.tensor1d(itemIndices, 'int32');
            
            // In-batch sampled softmax loss
            const loss = () => {
                const userEmbs = this.userForward(userTensor);
                const itemEmbs = this.itemForward(itemTensor, genreFeatures);
                
                // Compute similarity matrix: batch_size x batch_size
                const logits = tf.matMul(userEmbs, itemEmbs, false, true);
                
                // Labels: diagonal elements are positives
                const labels = tf.oneHot(
                    tf.range(0, userIndices.length, 1, 'int32'), 
                    userIndices.length
                );
                
                // Softmax cross entropy loss
                const loss = tf.losses.softmaxCrossEntropy(labels, logits);
                return loss;
            };
            
            // Compute gradients and update parameters
            const { value, grads } = this.optimizer.computeGradients(loss);
            
            this.optimizer.applyGradients(grads);
            
            return value.dataSync()[0];
        });
    }
    
    getUserEmbedding(userIndex) {
        return tf.tidy(() => {
            return this.userForward([userIndex]).squeeze();
        });
    }
    
    async getScoresForAllItems(userEmbedding) {
        return await tf.tidy(() => {
            // For inference, we need to compute item embeddings for all items
            // Since we don't have genre features for all items at once, we'll batch this
            const batchSize = 100;
            const numBatches = Math.ceil(this.numItems / batchSize);
            const allScores = [];
            
            for (let i = 0; i < numBatches; i++) {
                const start = i * batchSize;
                const end = Math.min(start + batchSize, this.numItems);
                const batchIndices = Array.from({length: end - start}, (_, j) => start + j);
                
                // Create dummy genre features for this batch (all zeros for now)
                const batchGenreFeatures = batchIndices.map(() => Array(this.numGenres).fill(0));
                
                const batchItemEmbs = this.itemForward(batchIndices, batchGenreFeatures);
                const batchScores = tf.dot(batchItemEmbs, userEmbedding);
                
                allScores.push(...batchScores.dataSync());
                
                // Clean up
                batchItemEmbs.dispose();
                batchScores.dispose();
            }
            
            return allScores;
        });
    }
    
    getItemEmbeddings() {
        // Return base item embeddings (without genre processing)
        return this.itemEmbeddings;
    }
}
