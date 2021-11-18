// Flores Vera Uriel

async function getData() {
    const datosCasasR = await fetch('https://static.platzi.com/media/public/uploads/datos-entrenamiento_15cd99ce-3561-494e-8f56-9492d4e86438.json');  
    const datosCasas = await datosCasasR.json();  
    const datosLimpios = datosCasas.map(casa => ({
      precio: casa.Precio,
      cuartos: casa.NumeroDeCuartosPromedio
    }))
    .filter(casa => (casa.precio != null && casa.cuartos != null));
    
    return datosLimpios;
  }
  
  function visualizarDatos(data){
    const valores = data.map(d => ({
      x: d.cuartos,
      y: d.precio,
    }));
  
    tfvis.render.scatterplot(
      {name: 'Cuartos vs Precio'},
      {values: valores}, 
      {
        xLabel: 'Cuartos',
        yLabel: 'Precio',
        height: 300
      }
    );
  }

function crearModelo(){
  const modelo = tf.sequential(); 
    
  // agregar capa oculta que va a recibir 1 dato
  modelo.add(tf.layers.dense({ inputShape: [1], units: 1, useBias: true }));
  
  // agregar una capa de salida que va a tener 1 sola unidad
  modelo.add(tf.layers.dense({ units: 1, useBias: true }));

  return modelo;
}


// Función de entrenamiento
// creacion de tres variables:
// - optimizador
// - perdida 
// - metricas


const optimizador = tf.train.adam()
const funcion_perdida = tf.losses.meanSquaredError;
const metricas = ['mse'];

async function entrenarModelo(model, inputs, labels) {
  // Prepare the model for training.  
  model.compile({
    optimizer: optimizador, 
    loss: funcion_perdida,
    metrics: metricas,
  });
  
  const surface = { name: 'show.history live', tab: 'Training' };
  const tamanioBatch = 28;
  const epochs = 50;
  const history = [];
  
  return await model.fit(inputs, labels, {
    tamanioBatch,
    epochs,
    shuffle: true,
    callbacks: tfvis.show.fitCallbacks(
      { name: 'Training Performance' },
      ['loss', 'mse'], 
      { height: 200, callbacks: ['onEpochEnd'] }
    )
  });
}



// Creación de modelo de regresion Lineal, por medio de dos MediaCapabilities, una oculta y una de salida
// -> TenserFlow prefiere que los datos esten normalizados, es decir que tenga valores entre 0 y 1.
// Proceso de limp ieza
// Empezará hacer un proceso de limpieza, en donde cada uno de esos datos se mezaclarán de manera aleatoria con un “shuffle”, para posteriormente ir mapeando cada uno de las entradas.
function convertirDatosATensores(data){
  return tf.tidy(() => {
    tf.util.shuffle(data);

    const entradas = data.map(d => d.cuartos)
    const etiquetas = data.map(d => d.precio);

    const tensorEntradas = tf.tensor2d(entradas, [entradas.length, 1]);
    const tensorEtiquetas = tf.tensor2d(etiquetas, [etiquetas.length, 1]);

    
    const entradasMax = tensorEntradas.max();
    const entradasMin = tensorEntradas.min();  
    const etiquetasMax = tensorEtiquetas.max();
    const etiquetasMin = tensorEtiquetas.min();


    // Creación de entradas normalizadas
    // (dato -min) / (max-min)
    // Funciones Primitivas de TensorFlow
    const entradasNormalizadas = tensorEntradas.sub(entradasMin).div(entradasMax.sub(entradasMin));
    const etiquetasNormalizadas = tensorEtiquetas.sub(etiquetasMin).div(etiquetasMax.sub(etiquetasMin));

      // Lo regresamos en un objeto, en donde estarán cada una de las variables, entradas normalizadas, etiquetas normalizadas, entradas Max, etc..
      return {
        entradas: entradasNormalizadas,
        etiquetas: etiquetasNormalizadas,
        // Return the min/max bounds so we can use them later.
        entradasMax,
        entradasMin,
        etiquetasMax,
        etiquetasMin,
      }

  });
}




// Función Run para probar que todo esté funcionando correctamente y que los datos sean los esperados, esta función recibe a data y lo almacena en una variable llamada tensor data.
// Entrenamiento de modelo --> Utiliza las API's de la clase anterior de entreamiento del modelo.

var modelo;
async function run() {

    const data = await getData();

    visualizarDatos(data);

    modelo = crearModelo();

    const tensorData = convertirDatosATensores(data);

    //Variables adicionales
    const {entradas, etiquetas} = tensorData;

    await entrenarModelo(modelo, entradas, etiquetas);
    
}

run();
