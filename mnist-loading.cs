using System;
using Microsoft.ML; // MLContext
using Microsoft.ML.Data; //VectorType
using System.IO; // Data pathing >:(


namespace Aphelios
{
    class Digit 
    // Represents one MNIST digit :)
    {
        [VectorType(785)] public float[] PixelValues;   
        // Each digit consists of 785 pixels
    }

    class DigitPrediction
    {
        public float[] Score;
    }
    class Program
    {
        private static string datapath = Path.Combine(Environment.CurrentDirectory, "handwritten_digits_large.csv");

    /***********************************************
        Main
        Parameters: args  --> cmd line arguments
    ***********************************************/
        static void Main(string[] args)
        {
            var context = new MLContext();
            Console.WriteLine("Loading data beepboop");

            var dataView = context.Data.LoadFromTextFile( // loads csv data directly into memory    
                path: datapath,
                columns: new[]
                {
                    new TextLoader.Column(nameof(Digit.PixelValues), DataKind.Single, 1, 784),
                    new TextLoader.Column("Number", DataKind.Single, 0)
                },
                hasHeader: false,
                separatorChar:',' // separator value 
            );


            // Partition data into training and test set 
            // Splits 20% for test data and 80% of the csv file for
            var partitions = context.Data.TrainTestSplit(dataView, testFraction: 0.2); // Data in place of MulticlassClassification
        
        
        }
    }
}
