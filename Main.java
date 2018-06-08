import weka.core.Instances;
import java.io.BufferedReader;
import java.io.FileReader;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.trees.J48;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.Evaluation;
import java.util.*;
import weka.core.*;
import weka.classifiers.*;
import weka.classifiers.evaluation.ThresholdCurve;
import java.util.List;
import weka.classifiers.meta.Bagging;
import weka.classifiers.meta.Vote;
import weka.classifiers.trees.RandomForest;


class HyperParameterTuner {

    public void tune() throws Exception {
        
        this.readData();
        this.classifier = this.createClassifier();
        List<String> optionList = this.getOptionList();
        double bestAuc = 0;
        double bestPRC = 0;
        for (String opt: optionList) {

            this.setOptions(weka.core.Utils.splitOptions(opt));

            this.classifier.buildClassifier(this.trainData);
            Evaluation eval = new Evaluation(this.trainData);
            eval.evaluateModel(this.classifier, this.testData);

            ThresholdCurve tc = new ThresholdCurve();
            int classIndex = 1;
            Instances result = tc.getCurve(eval.predictions(), classIndex);

            double auc = tc.getROCArea(result);
            double prc  = tc.getPRCArea(result);

            if (Math.abs(bestAuc - auc) > 0.00001 && bestAuc < auc) {
                bestAuc = auc;
                this.bestEval = eval;
                this.bestOption = opt;
                bestPRC = prc;
            } else if (Math.abs(bestAuc - auc) <= 0.00001) {
                
                if (bestPRC < prc) {
                    bestAuc = auc;
                    this.bestEval = eval;
                    this.bestOption = opt;
                    bestPRC = prc;
                }
            }
        }
    }

    public Evaluation getBestEvaluation() {
        return this.bestEval;
    }

    public String getScore() {
        Evaluation eval = this.getBestEvaluation();
        ThresholdCurve tc = new ThresholdCurve();
        int classIndex = 1;
        Instances result = tc.getCurve(eval.predictions(), classIndex);
        double accuracy = (eval.correct()/(double)this.testData.numInstances()) * 100;

        return Utils.doubleToString(eval.fMeasure(classIndex), 3)
            +"\t" + Utils.doubleToString(tc.getROCArea(result), 3) 
            + "\t" + Utils.doubleToString(tc.getPRCArea(result), 3)
            + "\t" + Utils.doubleToString(accuracy, 1) + "%";
    }

    public void printBest() throws Exception {

        Evaluation eval = this.getBestEvaluation();
        // generate curve
        ThresholdCurve tc = new ThresholdCurve();
        int classIndex = 1;
        Instances result = tc.getCurve(eval.predictions(), classIndex);

        System.out.println("Best options: " + this.bestOption);
        System.out.println(eval.toSummaryString("\nResults\n======\n", false));
        System.out.println("F-Measure: " + Utils.doubleToString(eval.fMeasure(classIndex), 3));
        System.out.println("AUC: " + Utils.doubleToString(tc.getROCArea(result), 3));
        System.out.println("PRC Area: " + Utils.doubleToString(tc.getPRCArea(result), 3));

        double accuracy = (eval.correct()/(double)this.testData.numInstances()) * 100;

        System.out.println("Best options: " + this.bestOption);
        System.out.println(Utils.doubleToString(eval.fMeasure(classIndex), 3)
            +"\t" + Utils.doubleToString(tc.getROCArea(result), 3) 
            + "\t" + Utils.doubleToString(tc.getPRCArea(result), 3)
            + "\t" + Utils.doubleToString(accuracy, 1) + "%");
    }

    public String getBestOption() {
        return this.bestOption;
    }

    protected void readData() {
        try {
            DataSource source = new DataSource(Main.TrainDatasetPath);
            Instances data = source.getDataSet();

            DataSource testSource = new DataSource(Main.TestDatasetPath);
            Instances test = testSource.getDataSet();

            if (data.classIndex() == -1)
                data.setClassIndex(data.numAttributes() - 1);

            if (test.classIndex() == -1)
                test.setClassIndex(test.numAttributes() - 1);


            this.trainData = data;
            this.testData = test;
        } catch (Exception e) {
            
        }
    }

    protected Classifier createClassifier() throws Exception {
        throw new Exception("Not implemented");
    }
    
    protected List<String> getOptionList() throws Exception {
        List<String> list = new ArrayList<String>();
        throw new Exception("Not implemented");
    }

    protected void setOptions(String[] options) throws Exception {
        throw new Exception("Not implemented");
    }

    private Instances trainData;
    private Instances testData;

    Classifier classifier;
    Evaluation bestEval;
    String bestOption;
}

class NBTuner extends HyperParameterTuner {
    protected Classifier createClassifier() {
        return new NaiveBayes();
    }

    protected List<String> getOptionList() {
        List<String> list = new ArrayList<String>();
        list.add("");
        list.add("-D");
        return list;
    }
    
    protected void setOptions(String[] options) throws Exception {
        NaiveBayes nb = (NaiveBayes) this.classifier;
        nb.setOptions(options);
    }
}

class BNTuner extends HyperParameterTuner {
    protected Classifier createClassifier() {
        return new BayesNet();
    }

    protected List<String> getOptionList() {
        List<String> list = new ArrayList<String>();
        list.add("-Q weka.classifiers.bayes.net.search.local.HillClimber -- -P 10 -mbc -S BAYES -E weka.classifiers.bayes.net.estimate.SimpleEstimator -- -A 0.5");
        return list;
    }
    
    protected void setOptions(String[] options) throws Exception {
        BayesNet nb = (BayesNet) this.classifier;
        nb.setOptions(options);
    }
}

class J48Tuner extends HyperParameterTuner {
    protected Classifier createClassifier() {
        return new J48();
    }

    protected List<String> getOptionList() {
        List<String> list = new ArrayList<String>();

        for (int i = 2; i < 20; ++i) {
            String opt = "-C 0.25 -M " + String.valueOf(i);
            list.add(opt);
        }

        return list;
    }
    
    protected void setOptions(String[] options) throws Exception {
        J48 j48 = (J48) this.classifier;
        j48.setOptions(options);
    }
}

class RandomForestTuner extends HyperParameterTuner {
    protected Classifier createClassifier() {
        return new RandomForest();
    }

    protected List<String> getOptionList() {
        List<String> list = new ArrayList<String>();

        for (int i = 20; i < 90; ++i) {
            String opt = "-P " + String.valueOf(i) + " -I 200 -num-slots 6 -K 0 -M 1.0 -V 0.001 -S 1";
            list.add(opt);
        }

        return list;
    }
    
    protected void setOptions(String[] options) throws Exception {
        RandomForest j48 = (RandomForest) this.classifier;
        j48.setOptions(options);
    }
}

class BaggingWithJ48Tuner extends HyperParameterTuner {

    protected Classifier createClassifier() {
        return new Bagging();
    }

    protected List<String> getOptionList() {
        List<String> list = new ArrayList<String>();

        for (int i = 60; i <= 150; i+=5) {
            for (int p = 20; p <= 90; p+=5) {
                String opt = String.format("-P %d -S 1 -num-slots 6 -I %d -W weka.classifiers.trees.J48 -- " + this.bestOption, p, i);
                list.add(opt);
            }
        }
        System.out.println("Option size:" + String.valueOf(list.size()));
        return list;
    }
    
    protected void setOptions(String[] options) throws Exception {
        
        Bagging bagging = (Bagging) this.classifier;
        bagging.setOptions(options);
    }

    public void setBaseOption(String option) {
        this.bestOption = option;
    }

    private String baseOption;
}

class BaggingWithNaiveBayes extends HyperParameterTuner {

    protected Classifier createClassifier() {
        return new Bagging();
    }

    protected List<String> getOptionList() {
        List<String> list = new ArrayList<String>();

        for (int i = 60; i <= 150; i+=5) {
            for (int p = 20; p <= 90; p+=5) {
                String opt = String.format("-P %d -S 1 -num-slots 6 -I %d -W weka.classifiers.bayes.NaiveBayes -- -D", p, i);
                list.add(opt);
            }
        }
        System.out.println("Option size:" + String.valueOf(list.size()));
        return list;
    }
    
    protected void setOptions(String[] options) throws Exception {
        Bagging bagging = (Bagging) this.classifier;
        bagging.setOptions(options);
    }
}

class BaggingWithBayesNet extends HyperParameterTuner {

    protected Classifier createClassifier() {
        return new Bagging();
    }

    protected List<String> getOptionList() {
        List<String> list = new ArrayList<String>();

        for (int i = 60; i <= 150; i+=5) {
            for (int p = 20; p <= 90; p+=5) {
                String opt = String.format("-P %d -S 1 -num-slots 6 -I %d -W weka.classifiers.bayes.BayesNet -- -Q weka.classifiers.bayes.net.search.local.HillClimber -- -P 10 -mbc -S BAYES -E weka.classifiers.bayes.net.estimate.SimpleEstimator -- -A 0.5", p, i);
                list.add(opt);
            }
        }
        System.out.println("Option size:" + String.valueOf(list.size()));
        return list;
    }
    
    protected void setOptions(String[] options) throws Exception {
        Bagging bagging = (Bagging) this.classifier;
        bagging.setOptions(options);
    }
}

class VoteTuner extends HyperParameterTuner {
    protected Classifier createClassifier() {
        return new Vote();
    }

    protected List<String> getOptionList() {

        String NB = "weka.classifiers.bayes.NaiveBayes " + Main.bestOptionForNB;
        String bayesNet = "weka.classifiers.bayes.BayesNet " + Main.bestOptionForBN;
        String j48 = "weka.classifiers.trees.J48 " + Main.bestOptionForJ48;
        String randomeForest = "weka.classifiers.trees.RandomForest " + Main.bestOptionForRandomForest;

        String baggingWithNB = "weka.classifiers.meta.Bagging " + Main.bestOptionForBaggingWithNB;
        String baggingWithBayesNet = "weka.classifiers.meta.Bagging " + Main.bestOptionForBaggingWithBayesNet;
        String baggingWithJ48 = "weka.classifiers.meta.Bagging " + Main.bestOptionForBaggingWithJ48;

        String[] classifers = {NB, bayesNet, j48, randomeForest, baggingWithNB, baggingWithBayesNet, baggingWithJ48};

        List<String> list = new ArrayList<String>();

        for (int mask = 1; mask < (1<<classifers.length); ++mask) {

            String option = "-S 1 ";

            int cnt = 0;
            for (int i = 0; i < classifers.length; ++i) {
                if ( (mask & (1<<i)) > 0 ) {
                    option += String.format("-B \"%s\" ", classifers[i]);
                    cnt++;
                }
            }
            if (cnt < 2) continue;

            // String[] voteMethods = {"-R PROD", "-R AVG", "-R MAJ", "-R MIN", "-R MAX"};
            String[] voteMethods = {"-R PROD", "-R AVG", "-R MIN"};

            for (String m: voteMethods) {
                // System.out.println(option + m);
                list.add(option + m);
            }
        }
        return list;
    }

    protected void setOptions(String[] options) throws Exception {
        Vote bagging = (Vote) this.classifier;
        bagging.setOptions(options);
    }
}

public class Main {

    public static String bestOptionForNB;
    public static String bestOptionForBN;
    public static String bestOptionForJ48;
    public static String bestOptionForBaggingWithNB = "-P 70 -S 1 -num-slots 6 -I 145 -W weka.classifiers.bayes.NaiveBayes -- -D";
    public static String bestOptionForBaggingWithBayesNet = "-P 70 -S 1 -num-slots 6 -I 90 -W weka.classifiers.bayes.BayesNet -- -Q weka.classifiers.bayes.net.search.local.HillClimber -- -P 10 -mbc -S BAYES -E weka.classifiers.bayes.net.estimate.SimpleEstimator -- -A 0.5";
    public static String bestOptionForBaggingWithJ48 = "-P 75 -S 1 -num-slots 6 -I 75 -W weka.classifiers.trees.J48 -- -C 0.25 -M 10";
    public static String bestOptionForRandomForest;
    public static String bestOptionForVote;
    
    static String finalTable = "";

    public static String TrainDatasetPath;
    public static String TestDatasetPath;

    static void nb() throws Exception {
        NBTuner t = new NBTuner();
        t.tune();
        t.printBest();

        Main.bestOptionForNB = t.getBestOption();
        Main.finalTable += t.getScore() + "\n";
    }

    static void bn() throws Exception {
        BNTuner t = new BNTuner();
        t.tune();
        t.printBest();
        Main.bestOptionForBN = t.getBestOption();
        Main.finalTable += t.getScore() + "\n";
    }

    static void j48() throws Exception {
        J48Tuner t = new J48Tuner();
        t.tune();
        t.printBest();
        Main.bestOptionForJ48 = t.getBestOption();
        Main.finalTable += t.getScore() + "\n";
    }

    static void randomeForest() throws Exception {
        RandomForestTuner t = new RandomForestTuner();
        t.tune();
        t.printBest();
        Main.bestOptionForRandomForest = t.getBestOption();
        Main.finalTable += t.getScore() + "\n";
    }

    static void baggingJ48() throws Exception {
        BaggingWithJ48Tuner t = new BaggingWithJ48Tuner();
        t.setBaseOption(Main.bestOptionForJ48);
        t.tune();
        t.printBest();
        Main.bestOptionForBaggingWithJ48 = t.getBestOption();
        Main.finalTable += t.getScore() + "\n";
    }

    static void baggingNaiveBayes() throws Exception {
        BaggingWithNaiveBayes t = new BaggingWithNaiveBayes();
        t.tune();
        t.printBest();
        Main.bestOptionForBaggingWithNB = t.getBestOption(); 
        Main.finalTable += t.getScore() + "\n";
    }

    static void baggingBayesNet() throws Exception {
        BaggingWithBayesNet t = new BaggingWithBayesNet();
        t.tune();
        t.printBest();
        Main.bestOptionForBaggingWithBayesNet = t.getBestOption();
        Main.finalTable += t.getScore() + "\n";
    }

    static void vote() throws Exception {
        VoteTuner t = new VoteTuner();
        t.tune();
        t.printBest();
        Main.finalTable += t.getScore() + "\n";
        Main.bestOptionForVote = t.getBestOption();
    }

    public static void main(String[] args) throws Exception {
        if (args.length != 2) {
            System.out.println(args.length);
            System.out.println("Usage: java -cp \".:./weka.jar\" Main TrainDataset TestDataset");
            return;
        }

        TrainDatasetPath = args[0];
        TestDatasetPath = args[1];
        System.out.println(String.format("Dealing with datasets: [%s] [%s]", TrainDatasetPath, TestDatasetPath));

        Main.finalTable = "";

        System.out.println("-------------------------Naive Bayes--------------------------");
        nb();
        System.out.println("-------------------------Naive Bayes-------------------------");

        System.out.println("\n----------------------------BayesNet----------------------------");
        bn();
        System.out.println("-------------------------------BayesNet----------------------------");

        System.out.println("\n--------------------------J48 Tree----------------------------");
        j48();
        System.out.println("----------------------------J48 Tree---------------------------");

        System.out.println("----------------------Random Forest-----------------");
        randomeForest();
        System.out.println("----------------------Random Forest-----------------");

        System.out.println("\n-------------------Bagging with Naive Bayes---------------------");
        baggingNaiveBayes();
        System.out.println("---------------------Bagging with Naive Bayes---------------------");

        System.out.println("\n--------------------------Bagging with BayesNet-------------");
        baggingBayesNet();
        System.out.println("---------------------Bagging with BayesNet---------------------");

        System.out.println("\n----Bagging with J48 Tree----");
        baggingJ48();
        System.out.println("---------------------Bagging with J48 Tree---------------------");

        System.out.println("\n----------------------------Vote------------------------------");
        vote();
        System.out.println("----------------------------Vote------------------------------");

        System.out.println("\n----------------------------Best Options------------------------------");
        String NB = "weka.classifiers.bayes.NaiveBayes " + Main.bestOptionForNB;
        String bayesNet = "weka.classifiers.bayes.BayesNet " + Main.bestOptionForBN;
        String j48 = "weka.classifiers.trees.J48 " + Main.bestOptionForJ48;
        String randomeForest = "weka.classifiers.trees.RandomForest " + Main.bestOptionForRandomForest;

        String baggingWithNB = "weka.classifiers.meta.Bagging " + Main.bestOptionForBaggingWithNB;
        String baggingWithBayesNet = "weka.classifiers.meta.Bagging " + Main.bestOptionForBaggingWithBayesNet;
        String baggingWithJ48 = "weka.classifiers.meta.Bagging " + Main.bestOptionForBaggingWithJ48;

        System.out.println(NB);
        System.out.println(bayesNet);
        System.out.println(j48);
        System.out.println(randomeForest);
        System.out.println(baggingWithNB);
        System.out.println(baggingWithBayesNet);
        System.out.println(baggingWithJ48);
        System.out.println("weka.classifiers.meta.Vote " + Main.bestOptionForVote);
        System.out.println("----------------------------Best Options------------------------------");

        System.out.println("\n----------------------------Score Table------------------------------");
        System.out.println(Main.finalTable);
        System.out.println("----------------------------Score Table------------------------------");
    }
}
