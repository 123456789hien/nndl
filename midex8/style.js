body {
  font-family: 'Poppins', sans-serif;
  background: linear-gradient(180deg, #e8f0ff, #ffffff);
  color: #04316e;
  margin: 0;
  padding: 0;
}

.container {
  width: 90%;
  max-width: 1100px;
  margin: 20px auto;
}

h1 {
  text-align: center;
  color: #003366;
}

.card {
  background: #fff;
  border-radius: 16px;
  box-shadow: 0 4px 12px rgba(0,0,0,0.1);
  padding: 20px;
  margin: 20px 0;
}

.hidden {
  display: none;
}

button {
  background-color: #0077cc;
  color: white;
  border: none;
  border-radius: 8px;
  padding: 10px 20px;
  cursor: pointer;
  transition: 0.3s;
}

button:hover {
  background-color: #005fa3;
}

input[type="file"], select, input[type="number"], input[type="range"] {
  margin: 10px;
  padding: 5px;
  border-radius: 6px;
  border: 1px solid #ccc;
}

#dataTable table, #statsTable table {
  width: 100%;
  border-collapse: collapse;
}

#dataTable th, #dataTable td, #statsTable th, #statsTable td {
  border: 1px solid #ddd;
  padding: 6px;
  text-align: center;
}

.insight-box {
  padding: 12px;
  border-radius: 10px;
  font-weight: bold;
  margin-top: 10px;
  text-align: center;
}

.insight-high { background-color: #ffcccc; color: #900; }
.insight-medium { background-color: #fff3cd; color: #856404; }
.insight-low { background-color: #d4edda; color: #155724; }
