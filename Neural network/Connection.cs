namespace NeuralNetwork
{
    //Connection from one node to another.
    [Serializable]
    public class Connection
	{
		public Node nodeIn;
		public Node nodeOut;
		public double weight;
		public double weightDerivative = 0d;
		public double weightVelocity = 0d;

		public Connection(Node nodeIn, Node nodeOut, double weight = 1d)
		{
			this.nodeIn = nodeIn;
			this.nodeOut = nodeOut;
			this.weight = weight;
		}

		/*
			UPDATE CONNECTION DATA.
		*/
		public void AddWeightDerivative()
		{

			weightDerivative += nodeOut.gradient * nodeIn.value;
        }

		public void UpdateWeight(double trainingStep, double momentum = 0.9d, double regularization = 0.1d)
		{
			double weightDecay = (1 - regularization * trainingStep); 
			double velocity = weightVelocity * momentum - weightDerivative * trainingStep;
			weightVelocity = velocity;
			weight = weight * weightDecay + velocity;
			weightDerivative = 0d;
		}

		/*
			CONNECTION WEIGHT INIT.
		*/
		public static double WeightInit(int prevLayerLength)
		{
			Random rand = new();
			double weight = RandomInNormalDistribution(rand, 0, 1) / Math.Sqrt(prevLayerLength);
			return weight;

            static double RandomInNormalDistribution(Random rng, double mean, double standardDeviation)
            {
                double x1 = 1 - rng.NextDouble();
                double x2 = 1 - rng.NextDouble();

                double y1 = Math.Sqrt(-2.0 * Math.Log(x1)) * Math.Cos(2.0 * Math.PI * x2);
                return y1 * standardDeviation + mean;
            }
        }
	}
}

