package analyzer;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;


import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap; 
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map; 
import java.util.Map.Entry;

import structures.Document;
import structures.Token;

enum FeatureSelectionMethod{ChiSquare,InformationGain}; 
public abstract class Classifier {//implements Optimizable.ByGradientValue {

	public double[] Parameters; 
	public int NumberOfProcessors;
	public HashMap<String, Token> vocabulary;
	public String[] vocabulary_array;
	public int NumberOfParameters;
	public Classifier(int NumberOfParameters,int NumberOfProcessors ,HashMap<String, Token> vocabulary )
	{
	 
		Parameters=new double[NumberOfParameters];
		this.NumberOfProcessors=NumberOfProcessors;
		this.vocabulary=vocabulary;
		this.vocabulary_array=vocabulary.keySet().toArray(new String[vocabulary.size()]);
		this.NumberOfParameters=NumberOfParameters;
	}
	
	public abstract double Classify(Document NewInstance );
	public abstract double Classify(Document NewInstance,double Threshold);
	public abstract void Train(ArrayList<Document> TrainingSet,double[] TrueLabels);
	public void PrintTrainingReview(Document r){
		String review="";
		for(String token:r.m_VSM.keySet())
			review+=token+":"+((double)Math.round(r.m_VSM.get(token) * 100) / 100)+",";
		System.out.println(review.substring(0, review.length()-1));
	}
	public void saveTopFeatures(String FileName){
		try {
			HashMap<Integer,Double> featureValues=new HashMap<Integer,Double>();
			
			for (int i=1;i<NumberOfParameters;++i)
				featureValues.put(i, Math.abs(Parameters[i]));
			// Sort
			List<Map.Entry<Integer,Double>> entries = new LinkedList<Map.Entry<Integer,Double>>(featureValues.entrySet());
			Collections.sort(entries, new Comparator<Map.Entry<Integer,Double>>() {
				@Override
				public int compare(Entry<Integer,Double> o1, Entry<Integer,Double> o2) {
					return o2.getValue().compareTo(o1.getValue());
				}
			});
			Map<Integer,Double> sortedMap = new LinkedHashMap<Integer,Double>();
			for(Map.Entry<Integer,Double> entry: entries){
				sortedMap.put(entry.getKey(), entry.getValue());
			}
			String[] vocab = vocabulary.keySet().toArray(new String[vocabulary.size()]);
			 
			
			FileWriter fstream = new FileWriter(FileName+".csv", false);
			BufferedWriter out = new BufferedWriter(fstream);
			String outstr="feature,value";
			for(int index:sortedMap.keySet())  
				outstr+="\n"+vocab[index-1]+","+Parameters[index];
			out.write(outstr);
			out.close();
			//System.out.println(FileName+" Classifier Saved!");
		} catch (Exception e) {
			e.printStackTrace(); 
		} 
	}
	public void Save(String FileName){
		try {
			FileWriter fstream = new FileWriter(FileName+".classifer", false);
			BufferedWriter out = new BufferedWriter(fstream);
			String outstr="";
			for(double param:Parameters)  
				outstr+=param+",";
			out.write(outstr.substring(0, outstr.length()-1));
			out.close();
			//System.out.println(FileName+" Classifier Saved!");
		} catch (Exception e) {
			e.printStackTrace(); 
		} 
	}
	public void Load(double[] Parameters){
	 
			for(int i=0;i<Parameters.length;++i)
				this.Parameters[i]=Parameters[i];
	 
	}
	public void Load(String FileName){
		try {
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream("./"+FileName+".classifer"), "UTF-8"));
			String[] params=reader.readLine().split(",");
			reader.close();
			for(int i=0;i<params.length;++i)
				Parameters[i]=Double.parseDouble(params[i]);
			//System.out.println(FileName+" Classifier Loaded!");
		} catch(IOException e){
			System.err.format("[Error]Failed to open file !!" );
		}
	}
}
