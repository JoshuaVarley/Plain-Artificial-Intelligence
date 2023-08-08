using System;
using System.Globalization;

namespace NeuralNetwork
{
    public class DataSetLoader
    {
        //FILE, A=LENGTH OF START VALUES THAT IS EXPECTED OUTPUT, arrayA= EXPECTED OUTPUT ARRAY, arrayB= INPUT ARRAY,  nodeDividerOutput=INPUT VALUE DIVIDER.
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
                            rowArrayB[i - A] = nodeValue / nodeDividerOutput;
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

