using System;
using System.Runtime.Serialization.Formatters.Binary;
using System.Runtime.Serialization;
using System.Xml;
using System.Xml.Serialization;

namespace NeuralNetwork
{
	public static class Serializer
	{
        public static class SerializerBinary
        {
            public static void SaveObjectToFile(string filePath, object obj)
            {
                #pragma warning disable SYSLIB0011
                try
                {
                    using (FileStream fs = new FileStream(filePath, FileMode.Create))
                    {
                        BinaryFormatter formatter = new BinaryFormatter();
                        formatter.Serialize(fs, obj);
                        Console.WriteLine("SAVED FILE SUCCESSFULLY.");
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"ERROR WHILE SAVING: {ex.Message}");
                }
            }

            public static T? LoadObjectFromFile<T>(string filePath)
            {
                #pragma warning disable SYSLIB0011
                try
                {
                    using (FileStream fs = new FileStream(filePath, FileMode.Open))
                    {
                        BinaryFormatter formatter = new BinaryFormatter();
                        var ob = (T?)formatter.Deserialize(fs);
                        Console.WriteLine("LOADED FILE SUCCESSFULLY.");
                        return ob;
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"ERROR WHILE LOADING: {ex.Message}");
                    return default(T?);
                }
            }
        }
    }
}

