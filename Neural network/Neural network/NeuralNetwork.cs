using System.Runtime.Serialization.Formatters.Binary;
using System.IO;
using System;
using System.Diagnostics;
using System.Threading;

namespace NeuralNetwork {
    [Serializable]
    public class Network {
		public Layer[] layers;

		/*
			INIT NETWORK.
		*/
		public Network(int[] layerLengths){
			//HEADER TEXT.
            Console.WriteLine($@"
______   ___   _____                            (
| ___ \ / _ \ |_   _|                       (   )  )
| |_/ // /_\ \  | |                          )  ( )
|  __/ |  _  |  | |                          .....
| |    | | | | _| |_                      .:::::::::.
\_|    \_| |_/ \___/                      ~\_______/~  (Yummy Pie)

PLAIN-ARTIFICIAL-INTELLIGENCE. CREATED BY ARE OLSEN, 01.08.2023.
-------------------------------------", Console.ForegroundColor = ConsoleColor.Magenta);

            this.layers = new Layer[layerLengths.Length];

			Console.WriteLine("INITIALIZING NETWORK STRUCTURE.");
			//INIT NODES.
			for(int i = 0; i < layerLengths.Length; i++){
				Node[] layerNodes = new Node[layerLengths[i]];
				for( int j = 0; j < layerLengths[i]; j++)
				{
					layerNodes[j] = new Node();
				}
				layers[i] = new Layer(layerNodes, (i==layerLengths.Length-1), (i==0));
			}

			//INIT CONNECTIONS.
			for(int i = layerLengths.Length-1; i >= 1 ; i--)
			{
				Layer layer = this.layers[i];
				Layer prevLayer = this.layers[i - 1];
				foreach (Node node in layer.nodes)
				{
					foreach (Node prevNode in prevLayer.nodes)
					{
						Connection con = new(prevNode, node, Connection.WeightInit(prevLayer.nodes.Length));
						node.inputConnections.Add(con);
						prevNode.outputConnections.Add(con);
					}
				}
			}
			Console.WriteLine("NETWORK STRUCTURE INITIALIZED.");
		}
		
		/*
			FORWARDPROPAGATION
		*/
		public double[] CalculateOutputs(double[] inputValues){
			layers[0].SetLayerValues(inputValues);
			for(int i = 1; i < layers.Length; i++){
				Layer layer = layers[i];
                layer.UpdateValues();
			}
			return layers[^1].GetLayerValues();
		}

		/*
			BACKPROPAGATION
		*/
		private void UpdateGradients(double[] expected){
            for (int i = layers.Length-1; i >= 1; i--){
				Layer layer = layers[i];
				layer.UpdateLayerGradients(expected);
            }
		}

		/*
			GRADIENT DESCENT ITERATION.
		*/
		private void ApplyGradients(double trainingStep, double momentum, double regularization)
		{
			for (int i = layers.Length - 1; i >= 1; i--)
			{
				foreach(Node node in layers[i].nodes)
				{
                    node.UpdateBias(trainingStep, momentum);
                    foreach (Connection connection in node.inputConnections)
                    {
						connection.UpdateWeight(trainingStep, momentum, regularization);
                    }
				}
			}
		}


		//Mini Batch Gradient Descent.
		private void MBGD(double trainingStep, double momentum, double regularization, double[][] batchInput, double[][] batchOutput)
		{
			//Multi-threading.
			/*Parallel.For(0, batchInput.Length, (i) =>
			{
                double[] sampleInput = batchInput[i];
                double[] sampleOutput = batchOutput[i];
                double[] calculates = CalculateOutputs(sampleInput);
                UpdateGradients(sampleOutput);
            });*/
			for(int i = 0; i < batchInput.Length; i++)
			{
                double[] sampleInput = batchInput[i];
                double[] sampleOutput = batchOutput[i];
                double[] calculates = CalculateOutputs(sampleInput);
                UpdateGradients(sampleOutput);
            }

			ApplyGradients(trainingStep/batchInput.Length, momentum, regularization);
		}

		public void Train(int epochs, int batch, double trainingStep, double momentum, double regularization, double[][] trainingDataInput, double[][] trainingDataOutput, double[][] testDataInput, double[][] testDataOutput)
		{
			Console.WriteLine("STARTING TRAINING.");
			for (int epochI = 0; epochI < epochs; epochI++)
			{
                Stopwatch stopwatch = new();
                stopwatch.Start();
                Console.WriteLine($"EPOCH_NUM : {epochI+1}");

				for(int iterationI = 0; iterationI+batch < trainingDataInput.Length; iterationI+=batch)
				{
					double[][] batchInput = new double[batch][];
					double[][] batchOutput = new double[batch][];
					for (int batchI = 0; batchI < batch; batchI++)
					{
						batchInput[batchI] = trainingDataInput[batchI + iterationI];
						batchOutput[batchI] = trainingDataOutput[batchI + iterationI];
					}

					MBGD(trainingStep, momentum, regularization, batchInput, batchOutput);
				}

                stopwatch.Stop();
				TimeSpan ts = stopwatch.Elapsed;
				Console.WriteLine($"EPOCH_TIME : {ts.Days}d, {ts.Hours}h, {ts.Minutes}m, {ts.Seconds}s");


                 Test(testDataInput, testDataOutput);
			}
			Console.WriteLine("TRAINING FINISHED.");
		}


        public void Test(double[][] testDataInput, double[][] testDataOutput)
		{
			double[][] calculated = new double[testDataInput.Length][];

			double count = 0d;

			for(int i = 0; i < calculated.Length; i++)
			{
				double[] outputs = CalculateOutputs(testDataInput[i]);
				calculated[i] = outputs;

				//Percentage correct.
				int outputIndex = Array.IndexOf(outputs, outputs.Max());
				int correctindex = Array.IndexOf(testDataOutput[i], testDataOutput[i].Max());
				if (outputIndex==correctindex)
				{
					count += 1d;
				}
			}

			double percentageCorrect = (count * 100) / (testDataOutput.Length);
			
			Console.WriteLine($@"
----------------------
MSE_AVG   : {Cost.MSE.CostMultipleFunction(testDataOutput, calculated)}
CROSS_AVG : {Cost.CROSS_ENTROPY.CostMultipleFunction(testDataOutput,calculated)}", Console.ForegroundColor = ConsoleColor.Magenta);
			Console.Write("CORRECT(%): ");
			Console.WriteLine($"{percentageCorrect}%", Console.ForegroundColor = (percentageCorrect<=60 ? ConsoleColor.Red: ((percentageCorrect<=90) ? ConsoleColor.Yellow  : ConsoleColor.Green)));
			Console.WriteLine("----------------------", Console.ForegroundColor=ConsoleColor.Magenta);
		}
	}
}