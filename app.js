class MovieLensApp {
    constructor() {
        this.interactions = [];
        this.items = new Map();
        this.userMap = new Map();
        this.itemMap = new Map();
        this.reverseUserMap = new Map();
        this.reverseItemMap = new Map();
        this.userTopRated = new Map();
        this.genreMap = new Map();
        this.genreList = [];
        
        this.basicModel = null;
        this.deepModel = null;
        
        this.config = {
            maxInteractions: 80000,
            embeddingDim: 32,
            batchSize: 512,
            epochs: 20,
            learningRate: 0.001
        };
        
        this.basicLossHistory = [];
        this.deepLossHistory = [];
        this.isTraining = false;
        
        this.initializeUI();
    }
    
    initializeUI() {
        document.getElementById('loadData').addEventListener('click', () => this.loadData());
        document.getElementById('train').addEventListener('click', () => this.train());
        document.getElementById('test').addEventListener('click', () => this.test());
        
        this.updateStatus('Click "Load Data" to start');
    }
    
    async loadData() {
        this.updateStatus('Loading data...');
        
        try {
            // Load interactions
            const interactionsResponse = await fetch('data/u.data');
            const interactionsText = await interactionsResponse.text();
            const interactionsLines = interactionsText.trim().split('\n');
            
            this.interactions = interactionsLines.slice(0, this.config.maxInteractions).map(line => {
                const parts = line.split('\t');
                // u.data format: user_id, item_id, rating, timestamp
                const [userId, itemId, rating, timestamp] = parts;
                return {
                    userId: parseInt(userId),
                    itemId: parseInt(itemId),
                    rating: parseFloat(rating),
                    timestamp: parseInt(timestamp)
                };
            });
            
            // Load items and genres
            const itemsResponse = await fetch('data/u.item');
            const itemsText = await itemsResponse.text();
            const itemsLines = itemsText.trim().split('\n');
            
            // Parse genre list (last 19 fields in u.item)
            this.genreList = [
                "Unknown", "Action", "Adventure", "Animation", "Children's", 
                "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
                "Film-Noir", "Horror", "Musical", "Mystery", "Romance",
                "Sci-Fi", "Thriller", "War", "Western"
            ];
            
            itemsLines.forEach(line => {
                const parts = line.split('|');
                if (parts.length < 5) return; // Skip invalid lines
                
                const itemId = parseInt(parts[0]);
                let title = parts[1] || '';
                let year = null;
                
                // Extract year from title - handle various title formats
                if (title) {
                    const yearMatch = title.match(/\((\d{4})\)/);
                    if (yearMatch) {
                        year = parseInt(yearMatch[1]);
                        // Clean title by removing the year
                        title = title.replace(/\(\d{4}\)/, '').trim();
                    }
                }
                
                // Parse genre flags (last 19 fields)
                // u.item format: movie id | movie title | release date | video release date | IMDb URL | genre1|genre2|...|genre19
                const genreFlags = parts.slice(5, 24).map(flag => {
                    const num = parseInt(flag);
                    return isNaN(num) ? 0 : num;
                });
                
                // Ensure we have exactly 19 genre flags
                const paddedGenres = Array(19).fill(0);
                genreFlags.forEach((flag, index) => {
                    if (index < 19) paddedGenres[index] = flag;
                });
                
                this.items.set(itemId, {
                    title: title,
                    year: year,
                    genres: paddedGenres
                });
                
                // Store genre mapping for quick access
                this.genreMap.set(itemId, paddedGenres);
            });
            
            // Create mappings and find users with sufficient ratings
            this.createMappings();
            this.findQualifiedUsers();
            
            this.updateStatus(`Loaded ${this.interactions.length} interactions and ${this.items.size} items. ${this.qualifiedUsers.length} users have 20+ ratings. ${this.genreList.length} genres detected.`);
            
            document.getElementById('train').disabled = false;
            
        } catch (error) {
            this.updateStatus(`Error loading data: ${error.message}`);
            console.error('Detailed error:', error);
        }
    }
    
    createMappings() {
        // Create user and item mappings to 0-based indices
        const userSet = new Set(this.interactions.map(i => i.userId));
        const itemSet = new Set(this.interactions.map(i => i.itemId));
        
        Array.from(userSet).forEach((userId, index) => {
            this.userMap.set(userId, index);
            this.reverseUserMap.set(index, userId);
        });
        
        Array.from(itemSet).forEach((itemId, index) => {
            this.itemMap.set(itemId, index);
            this.reverseItemMap.set(index, itemId);
        });
        
        // Group interactions by user and find top rated movies
        const userInteractions = new Map();
        this.interactions.forEach(interaction => {
            const userId = interaction.userId;
            if (!userInteractions.has(userId)) {
                userInteractions.set(userId, []);
            }
            userInteractions.get(userId).push(interaction);
        });
        
        // Sort each user's interactions by rating (desc) and timestamp (desc)
        userInteractions.forEach((interactions, userId) => {
            interactions.sort((a, b) => {
                if (b.rating !== a.rating) return b.rating - a.rating;
                return b.timestamp - a.timestamp;
            });
        });
        
        this.userTopRated = userInteractions;
    }
    
    findQualifiedUsers() {
        // Filter users with at least 20 ratings
        const qualifiedUsers = [];
        this.userTopRated.forEach((interactions, userId) => {
            if (interactions.length >= 20) {
                qualifiedUsers.push(userId);
            }
        });
        this.qualifiedUsers = qualifiedUsers;
    }
    
    // ... rest of the methods remain the same ...
    async train() {
        if (this.isTraining) return;
        
        this.isTraining = true;
        document.getElementById('train').disabled = true;
        this.basicLossHistory = [];
        this.deepLossHistory = [];
        
        this.updateStatus('Initializing models...');
        
        // Initialize both models
        this.basicModel = new BasicTwoTowerModel(
            this.userMap.size,
            this.itemMap.size,
            this.config.embeddingDim
        );
        
        this.deepModel = new DeepTwoTowerModel(
            this.userMap.size,
            this.itemMap.size,
            this.config.embeddingDim,
            this.genreList.length
        );
        
        // Prepare training data
        const userIndices = this.interactions.map(i => this.userMap.get(i.userId));
        const itemIndices = this.interactions.map(i => this.itemMap.get(i.itemId));
        
        this.updateStatus('Starting training for both models...');
        
        // Training loop for both models
        const numBatches = Math.ceil(userIndices.length / this.config.batchSize);
        
        for (let epoch = 0; epoch < this.config.epochs; epoch++) {
            let basicEpochLoss = 0;
            let deepEpochLoss = 0;
            
            for (let batch = 0; batch < numBatches; batch++) {
                const start = batch * this.config.batchSize;
                const end = Math.min(start + this.config.batchSize, userIndices.length);
                
                const batchUsers = userIndices.slice(start, end);
                const batchItems = itemIndices.slice(start, end);
                
                // Get genre features for deep model
                const batchGenreFeatures = batchItems.map(itemIdx => {
                    const itemId = this.reverseItemMap.get(itemIdx);
                    return this.genreMap.get(itemId) || Array(this.genreList.length).fill(0);
                });
                
                // Train both models
                const basicLoss = await this.basicModel.trainStep(batchUsers, batchItems);
                const deepLoss = await this.deepModel.trainStep(batchUsers, batchItems, batchGenreFeatures);
                
                basicEpochLoss += basicLoss;
                deepEpochLoss += deepLoss;
                
                this.basicLossHistory.push(basicLoss);
                this.deepLossHistory.push(deepLoss);
                this.updateLossChart();
                
                if (batch % 10 === 0) {
                    this.updateStatus(`Epoch ${epoch + 1}/${this.config.epochs}, Batch ${batch}/${numBatches}, Basic Loss: ${basicLoss.toFixed(4)}, Deep Loss: ${deepLoss.toFixed(4)}`);
                }
                
                // Allow UI to update
                await new Promise(resolve => setTimeout(resolve, 0));
            }
            
            basicEpochLoss /= numBatches;
            deepEpochLoss /= numBatches;
            this.updateStatus(`Epoch ${epoch + 1}/${this.config.epochs} completed. Basic Avg Loss: ${basicEpochLoss.toFixed(4)}, Deep Avg Loss: ${deepEpochLoss.toFixed(4)}`);
        }
        
        this.isTraining = false;
        document.getElementById('train').disabled = false;
        document.getElementById('test').disabled = false;
        
        this.updateStatus('Training completed! Click "Test" to compare recommendations.');
        
        // Visualize embeddings from basic model
        this.visualizeEmbeddings();
    }

    updateLossChart() {
        const canvas = document.getElementById('lossChart');
        const ctx = canvas.getContext('2d');
        
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        if (this.basicLossHistory.length === 0) return;
        
        const allLosses = [...this.basicLossHistory, ...this.deepLossHistory];
        const maxLoss = Math.max(...allLosses);
        const minLoss = Math.min(...allLosses);
        const range = maxLoss - minLoss || 1;
        
        // Draw basic model loss
        ctx.strokeStyle = '#007acc';
        ctx.lineWidth = 2;
        ctx.beginPath();
        
        this.basicLossHistory.forEach((loss, index) => {
            const x = (index / this.basicLossHistory.length) * canvas.width;
            const y = canvas.height - ((loss - minLoss) / range) * canvas.height;
            
            if (index === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        });
        
        ctx.stroke();
        
        // Draw deep model loss
        ctx.strokeStyle = '#28a745';
        ctx.lineWidth = 2;
        ctx.beginPath();
        
        this.deepLossHistory.forEach((loss, index) => {
            const x = (index / this.deepLossHistory.length) * canvas.width;
            const y = canvas.height - ((loss - minLoss) / range) * canvas.height;
            
            if (index === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        });
        
        ctx.stroke();
        
        // Add labels
        ctx.fillStyle = '#000';
        ctx.font = '12px Arial';
        ctx.fillText(`Min: ${minLoss.toFixed(4)}`, 10, canvas.height - 10);
        ctx.fillText(`Max: ${maxLoss.toFixed(4)}`, 10, 20);
    }

    // ... other methods remain the same ...
}

// Initialize app when page loads
let app;
document.addEventListener('DOMContentLoaded', () => {
    app = new MovieLensApp();
});
