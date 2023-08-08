using System.Globalization;
namespace NeuralNetwork
{
	public class Program
	{
		public static void Main(string[] args)
		{
			//INIT NETWORK.
			Network? network = new Network(new int[] { 784, 100, 10 });

			//LOAD DATA FROM MNIST INTO ARRAYS.
			double[][] trainInput;
			double[][] trainOutput;

            double[][] testInput;
            double[][] testOutput;
            DataSetLoader.LoadCsvToArray("../../../mnist_train.csv", 1, out trainOutput, out trainInput, 255d);
            DataSetLoader.LoadCsvToArray("../../../mnist_test.csv", 1, out testOutput, out testInput, 255d);

			//FIX OUTPUT ARRAY, (DATASET DOESN'T USE ARRAY FOR OUTPUT, BUT 1 NUMBER :| ).
            for (int i = 0; i < trainOutput.Length; i++)
			{
				double[] temp = new double[10] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
				temp[Convert.ToInt32(trainOutput[i][0])] = 1d;
				trainOutput[i] = temp;
			}

			for (int i = 0; i < testOutput.Length; i++)
			{
				double[] temp = new double[10] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
				temp[Convert.ToInt32(testOutput[i][0])] = 1d;
				testOutput[i] = temp;
			}

			//TRAIN NETWORK.
			network.Train(8, 32, 0.05d, 0.9d, 0.1d, trainInput, trainOutput, testInput, testOutput);

			//SAVE NETWORK.
			Serializer.SerializerBinary.SaveObjectToFile("../../../networkserialized", network);
            //network = Serializer.SerializerBinary.LoadObjectFromFile<Network>("../../../networkserialized");
            //network.Test(testInput, testOutput);
            //network.Test(testInput, testOutput);
        }
	}
}

