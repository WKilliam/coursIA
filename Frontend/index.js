const express = require('express');
const axios = require('axios');
const path = require('path');
const app = express();
const PORT = 3000;

app.use(express.urlencoded({ extended: true }));
app.use(express.json());

// Utilisation de 'path.join' pour spécifier le dossier contenant les ressources statiques
app.use(express.static(path.join(__dirname, 'public')));

// Route pour afficher le contenu de index.html
app.get('/dashboard', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});


// Route pour effectuer une requête HTTP
app.get('/api-data', async (req, res) => {
    try {
        const response = await axios.get('URL_DE_VOTRE_API'); // Remplacez 'URL_DE_VOTRE_API' par l'URL de votre API
        const apiData = response.data;

        // Traitez les données obtenues de votre API comme nécessaire
        // ...

        res.json(apiData); // Renvoie les données traitées au format JSON
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

app.listen(PORT, () => {
    console.log(`Server running on http://localhost:${PORT}`);
});
