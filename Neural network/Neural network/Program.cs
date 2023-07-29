using System.Globalization;

namespace NeuralNetwork
{
	public class Program
	{
		public static void Main(string[] args)
		{
			var network = new Network(new int[] { 784, 250, 50, 10 });
 
			double[][] trainInput;
			double[][] trainOutput;
			DataSetLoader.LoadCsvToArray("../../../mnist_train.csv", 1, out trainOutput, out trainInput, 255d);

			for (int i = 0; i < trainOutput.Length; i++)
			{
				double[] temp = new double[10] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
				temp[Convert.ToInt32(trainOutput[i][0])] = 1d;
				trainOutput[i] = temp;
			}

			double[][] testInput;
			double[][] testOutput;
			DataSetLoader.LoadCsvToArray("../../../mnist_test.csv", 1, out testOutput, out testInput, 255d);
			for (int i = 0; i < testOutput.Length; i++)
			{
				double[] temp = new double[10] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
				temp[Convert.ToInt32(testOutput[i][0])] = 1d;
				testOutput[i] = temp;
			}

			Console.WriteLine("Training...");
			network.Train(1, 1000, 128, 0.1d, 0.9d, trainInput, trainOutput);
			Console.WriteLine("Testing...");
			network.Test(testInput, testOutput);
		}
	}






	public class DataSetLoader
	{
		public static void LoadCsvToArray(string filePath, int A, out double[][] arrayA, out double[][] arrayB, double nodeDividerOutput = 1d)
		{
			List<double[]> tempArrayA = new List<double[]>();
			List<double[]> tempArrayB = new List<double[]>();

			using (var reader = new StreamReader(filePath))
			{
				string? line;
				int lineCount = 0;

				while ((line = reader.ReadLine()) != null)
				{
					if (lineCount == 0)
					{
						lineCount++;
						continue; // Skip the first row, usually just labels.
					}

					string[] values = line.Split(',');

					if (values.Length < A)
					{
						Console.WriteLine($"Warning: Line {lineCount + 1} contains less than {A} elements.");
						continue;
					}

					double[] rowArrayA = new double[A];
					double[] rowArrayB = new double[values.Length - A];

					for (int i = 0; i < values.Length; i++)
					{
						double nodeValue;

						if (!double.TryParse(values[i], NumberStyles.Float, CultureInfo.InvariantCulture, out nodeValue))
						{
							Console.WriteLine($"Error: Line {lineCount}, Element {i} is not a valid double value.");
							continue;
						}

						if (i < A)
							rowArrayA[i] = nodeValue;
						else
							rowArrayB[i - A] = nodeValue/nodeDividerOutput;
					}

					tempArrayA.Add(rowArrayA);
					tempArrayB.Add(rowArrayB);

					lineCount++;
				}
			}

			arrayA = tempArrayA.ToArray();
			arrayB = tempArrayB.ToArray();
		}
	}
}

