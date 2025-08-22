import express from 'express';
import apiRoutes from './routes/api';

const app = express();
const port = process.env.PORT || 3000;

app.use(express.json());

app.use('/api', apiRoutes);

app.get('/', (req, res) => {
  res.send('Hello from the backend!');
});

app.listen(port, () => {
  console.log(`Backend server is running on http://localhost:${port}`);
});
