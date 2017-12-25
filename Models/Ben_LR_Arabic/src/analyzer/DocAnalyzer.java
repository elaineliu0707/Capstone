/**
 * 
 */
package analyzer;


import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream; 
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;  
import java.text.DateFormat;
import java.text.SimpleDateFormat; 
import java.util.Arrays;
import java.util.Date; 
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map; 
import java.util.Random;
import java.util.Set; 
import java.util.Map.Entry;
import java.util.TreeMap; 
 





import org.apache.commons.io.FilenameUtils;  

import structures.Document;
import structures.Token;
 
import edu.stanford.nlp.international.arabic.process.*;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.process.Tokenizer;
import edu.stanford.nlp.process.TokenizerFactory; 
public class DocAnalyzer   {
	public static void main(String[] args) { 
		
		DocAnalyzer docAnalyzer=new DocAnalyzer(8,true);  
		 docAnalyzer.RunAnalyzerOnOneGroup( "C:\\Users\\Mohammad\\Desktop\\arabic-docs", "C:\\Users\\Mohammad\\Desktop\\arabic-docs-labels.txt" ,new String[]{},"hezbollah");
	   
	  } 
  
	public static DateFormat dateFormat = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss");
	//ArrayList<User> Users;
	ArrayList<Document> trainingDocuments,testingDocuments;
	public ArrayList<TokenizerFactory<CoreLabel>> tokenizer; // need many because of the threading
	//a list of stopwords
	HashSet<String> m_stopwords; 
	public Boolean debug=false;
	Random RandomGen;
	//you can store the loaded reviews in this arraylist for further processing
	//ArrayList<Post> m_reviews;
	int documentsCount;
	//you might need something like this to store the counting statistics for validating Zipf's and computing IDF
	public HashMap<String, Token> m_Vocabs;	
	public int TotalPos;
	public int TotalNeg;
	private Object lock1 = new Object();
	private Object lock2 = new Object();
	//	public Object lockA,lockB,lockC,lockD;
	private int MaxTokenID;
	//we have also provided sample implementation of language model in src.structures.LanguageModel
	private int NumberOfProcessors;
	public void init(){
		TotalPos=0;
		TotalNeg=0; 
		trainingDocuments=new ArrayList<Document>();
		testingDocuments=new ArrayList<Document>();
		m_Vocabs=new HashMap<String, Token>();
		MaxTokenID=0;
		documentsCount=0; 
	}
	public DocAnalyzer(int NumberOfProcessors ,Boolean debug) {
		this.debug=debug;
		this.NumberOfProcessors=NumberOfProcessors; 
		RandomGen=new Random();
		init();
		m_stopwords= new HashSet<String>();
		tokenizer=new ArrayList<TokenizerFactory<CoreLabel>>();
		for(int i=0;i<NumberOfProcessors;++i)
		{
			TokenizerFactory<CoreLabel> tf= ArabicTokenizer.atbFactory() ;
			String[] parameters =
		            new String[]
		            {
		                "normArDigits", "normAlif",  "removeDiacritics", "removeTatweel", "removeProMarker",
		                "removeSegMarker", "removeMorphMarker", "removeLengthening", "atbEscaping"
		            };
			 for  (String option : parameters)
		     {tf.setOptions(option);} 
		    tokenizer.add(tf);
		}
		// Load Stopwards
		LoadStopwords("data/arabic.stop");
	}


	 
	//sample code for loading a list of stopwords from file
	//you can manually modify the stopword file to include your newly selected words
	public void LoadStopwords(String filename) {
		TokenizerFactory<CoreLabel> tf= ArabicTokenizer.atbFactory() ;
		String[] parameters =
		        new String[]
		        {
		            "normArDigits", "normAlif",  "removeDiacritics", "removeTatweel", "removeProMarker",
		            "removeSegMarker", "removeMorphMarker", "removeLengthening", "atbEscaping"
		        };
		 for  (String option : parameters)
		 {tf.setOptions(option);} 
		 
		 
		// Read the file  
		final String encoding = "UTF-8"; 
		try { 
		  Tokenizer<CoreLabel> tokenizer = tf.getTokenizer(new InputStreamReader(new FileInputStream(filename), encoding),"untokenizable=noneDelete");
		
		  while (tokenizer.hasNext()) {  
		    String word = NormalizationDemo(tokenizer.next().word()); 
		    if (!word.isEmpty())
				m_stopwords.add(word);
		  } 
		} catch (Exception e) { 
		  e.printStackTrace(); 
		} 
		 
		if(debug)
			System.out.format("Loading %d stopwords from %s\n", m_stopwords.size(), filename);
	}

	public void analyzeDocumentDemo(String filename,int isPositive, int core) {		
		try {

			 
			// process Content
			ArrayList<String> AddedTokens=new ArrayList<String>();
			String previousToken="";
			Tokenizer<CoreLabel> tokenizer = this.tokenizer.get(core).getTokenizer(new InputStreamReader(new FileInputStream(filename), "utf-8"),"untokenizable=noneDelete");
			 while (tokenizer.hasNext()) { 
			          
			        String finalToken = NormalizationDemo(tokenizer.next().word()); 
			       
				if(!finalToken.isEmpty()) // if the token is empty, then try next token
				{ 


					// add uni-grams and bigrams to the hashmap.

					synchronized(lock1) {


						// unigram
						if(!m_Vocabs.containsKey(finalToken)&&!m_stopwords.contains(finalToken)) 
							m_Vocabs.put(finalToken, new Token(MaxTokenID++,finalToken));

						if(m_Vocabs.containsKey(finalToken)	&&!AddedTokens.contains(finalToken) ){
							m_Vocabs.get(finalToken).setValue( m_Vocabs.get(finalToken).getValue()+1);// increase count
							if(isPositive==1)
								m_Vocabs.get(finalToken).setPosDF( m_Vocabs.get(finalToken).getPosDF()+1);// increase count
							else
								m_Vocabs.get(finalToken).setNegDF( m_Vocabs.get(finalToken).getNegDF()+1);
							AddedTokens.add(finalToken);}
						// bigram
						String bigram=previousToken+"-"+finalToken;
						if(!previousToken.isEmpty()&&!m_Vocabs.containsKey(bigram)&&!(m_stopwords.contains(previousToken)&&m_stopwords.contains(finalToken)))
							m_Vocabs.put(bigram, new Token(MaxTokenID++,bigram));
						if(m_Vocabs.containsKey(bigram)&&!AddedTokens.contains(bigram)){
							m_Vocabs.get(bigram).setValue( m_Vocabs.get(bigram).getValue()+1);// increase count
							if(isPositive==1)
								m_Vocabs.get(bigram).setPosDF( m_Vocabs.get(bigram).getPosDF()+1);// increase count
							else
								m_Vocabs.get(bigram).setNegDF( m_Vocabs.get(bigram).getNegDF()+1);
							AddedTokens.add(bigram);
						}

					}
				}
				previousToken=finalToken;
			}
			synchronized(lock2) {
				documentsCount++;
				if(isPositive==1)
					TotalPos++;
				else
					TotalNeg++;
			}

		} catch(IOException e){
			System.err.format("[Error]Failed to open file %s!!", filename);
		}

	}

	public void analyzeVSM(String filename,int isPositive,int NumberOfDocumentsInTraining,int core,  Boolean training ) {		
		try {

			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
			String line; 
			Document doc= new Document();
			String content="";
			while ((line = reader.readLine()) != null&&!line.isEmpty()) {
				content+=" "+line;
			}
			reader.close();
			doc.setContent(content);
			doc.setLabel(isPositive);
			doc.setFileName(filename);
			// process Content
			String previousToken="";
			Tokenizer<CoreLabel> tokenizer = this.tokenizer.get(core).getTokenizer(new InputStreamReader(new FileInputStream(filename), "UTF-8"),"untokenizable=noneDelete");
			 while (tokenizer.hasNext()) {  
			        String finalToken = NormalizationDemo(tokenizer.next().word());  
				if(!finalToken.isEmpty()) // if the token is empty, then try next token
				{ 



					// add uni-grams and bigrams to the hashmap.
					synchronized(lock1) {
						if(m_Vocabs.containsKey(finalToken)){
							String vocabID=finalToken;
							if(!doc.m_VSM.containsKey(vocabID))
								doc.m_VSM.put(vocabID, 0.0);
							doc.m_VSM.put(vocabID,doc.m_VSM.get(vocabID)+1);// increase count

						}
						String bigram=previousToken+"-"+finalToken;
						// bigram
						if(m_Vocabs.containsKey(bigram)){
							if(!doc.m_VSM.containsKey(bigram))
								doc.m_VSM.put(bigram, 0.0);
							doc.m_VSM.put(bigram,doc.m_VSM.get(bigram)+1);// increase count

						}
					}
				}
				previousToken=finalToken;
			}
			if(doc.m_VSM.size()<=5)// empty vector or less than 5 features .. do not add 
				return;
			// normalize TF (Sub-linear TF scaling) and them multiply by IDF to obtain TF-IDF
			Set<String> set = doc.m_VSM.keySet();
			Iterator<String> itr = set.iterator();
			while (itr.hasNext())
			{
				String key = itr.next();
				doc.m_VSM.put(key,(1+Math.log10(doc.m_VSM.get(key)))*(1+Math.log10(NumberOfDocumentsInTraining/m_Vocabs.get(key).getValue())));
			}
			doc.CalculateNorm();
			doc.clearContent();



			synchronized(lock2) { 
				if(training)
					trainingDocuments.add(doc);
				else
					testingDocuments.add(doc);
			}

		} catch(IOException e){
			System.err.format("[Error]Failed to open file %s!!", filename);
		}

	}

	public ArrayList<String> GetFiles(String folder, String suffix) {
		if(debug)
			System.out.println(folder);
		File dir = new File(folder);
		ArrayList<String> Files=new ArrayList<String>();
		for (File f : dir.listFiles()) {
			if (f.isFile() && f.getName().endsWith(suffix)){
				Files.add(f.getAbsolutePath());
			}
			else if (f.isDirectory())
				Files.addAll(GetFiles(f.getAbsolutePath(), suffix)) ;
		}
		return Files;
	}
	//sample code for demonstrating how to perform text normalization
	public String NormalizationDemo(String token) {
		// convert to lower case
		token = token.toLowerCase();
		//token = token.replaceAll("\\d+star(s)?", "RATE");// rating by stars
		// Some scales and measures
		//token = token.replaceAll("\\d+(oz|lb|lbs|cent|inch|piec)", "SCALE");
		// convert some of the dates/times formats
		//token = token.replaceAll("\\d{2}(:\\d{2})?(\\s)?(a|p)m", "TIME"); // 12 hours format
		//token = token.replaceAll("\\d{2}:\\d{2}", "TIME"); // 24 hours format
		//token = token.replaceAll("\\d{1,2}(th|nd|st|rd)", "DATE");// 1st 2nd 3rd 4th date format
		// convert numbers
		token = token.replaceAll("\\d+.\\d+", "NUM");		
		token = token.replaceAll("\\d+(ish)?", "NUM");
		// tested on "a 123 b 3123 c 235.123 d 0 e 0.3 f 213.231.1321 g +123 h -123.123"
		// remove punctuations
		token = token.replaceAll("–", ""); 
		token = token.replaceAll("’", ""); 
		token = token.replaceAll("“", ""); 
		token = token.replaceAll("”", "");  
		token = token.replaceAll("\\p{Punct}", ""); 
		//tested on this string:  "This., -/ is #! an <>|~!@#$%^&*()_-+=}{[]\"':;?/>.<, $ % ^ & * example ;: {} of a = -_ string with `~)() punctuation" 
		return token;
	}

	public void RemoveVocabTail()
	{
		Set<String> set = m_Vocabs.keySet();
		Iterator<String> itr = set.iterator();
		while (itr.hasNext())
		{
			String key = itr.next();
			if (m_Vocabs.get(key).getValue()<50)  
				itr.remove(); 
		}
	}
	public void Save(HashMap<String,Token> Map,String Type)
	{

		// Sort
		ArrayList<Token> sortedTokens = new ArrayList<Token>(Map.values());
		Collections.sort(sortedTokens, new Comparator<Token>() {

			public int compare(Token T1, Token T2) {
				return Double.compare(T2.getValue(),T1.getValue());
			}
		});
		// Save to csv file
		try {
			FileWriter fstream = new FileWriter("./"+Type+".csv", false);
			BufferedWriter out = new BufferedWriter(fstream);
			Iterator<Token> iter = sortedTokens.iterator();
			while (iter.hasNext())  
			{
				Token t=iter.next();
				out.write(t.getID()+","+t.getToken() + ","+t.getValue()+ ","+t.getPosDF()+ ","+t.getNegDF()+"\n");
			}
			out.close();
			if(debug)
				System.out.println(Type+" Saved!");
		} catch (Exception e) {
			e.printStackTrace(); 
		}

	}
	public void Save(ArrayList<Token> sortedTokens,String Type)
	{

		// Sort
		Collections.sort(sortedTokens, new Comparator<Token>() {
			public int compare(Token T1, Token T2) {
				return Double.compare(T1.getValue(),T2.getValue());
			}
		});
		// Save to csv file
		try {
			FileWriter fstream = new FileWriter("./"+Type+".csv", false);
			BufferedWriter out = new BufferedWriter(fstream);
			Iterator<Token> iter = sortedTokens.iterator();
			while (iter.hasNext())  
			{
				Token t=iter.next();
				out.write(t.getID()+","+t.getToken() + ","+t.getValue()+ ","+t.getPosDF()+ ","+t.getNegDF()+"\n");
			}
			out.close();
			if(debug)
				System.out.println(Type+" Saved!");
		} catch (Exception e) {
			e.printStackTrace(); 
		}

	}

	public HashMap<String, Integer> GetLabels(String filename) {

		HashMap<String, Integer> labels = new HashMap<String, Integer>();
		try {
			BufferedReader reader = new BufferedReader(new InputStreamReader(
					new FileInputStream(filename), "UTF-8"));
			String line;

			while ((line = reader.readLine()) != null) {
				// it is very important that you perform the same processing
				// operation to the loaded stopwords
				// otherwise it won't be matched in the text content
				String[] vals = line.split(",");
				if (!line.isEmpty())
					labels.put(vals[0], Integer.parseInt(vals[1]));
			}
			reader.close();
		} catch (IOException e) {
			System.err.format("[Error]Failed to open file %s!!", filename);
		}
		return labels;
	}
	public ArrayList<ArrayList<String>> LoadCVTestingDocuments(String filename) {
		ArrayList<ArrayList<String>> docs = new  ArrayList<ArrayList<String>>();
		try {
			BufferedReader reader = new BufferedReader(new InputStreamReader(
					new FileInputStream(filename), "UTF-8"));
			String line;

			while ((line = reader.readLine()) != null) { 
				String[] vals = line.split(" ");
				if (!line.isEmpty())
					docs.add(new ArrayList<String>(Arrays.asList(vals)));
			}
			reader.close();
		} catch (IOException e) {
			System.err.format("[Error]Failed to open file %s!!", filename);
		}
		return docs;
	}
	public void BuildVocab(ArrayList<String>Files,HashMap<String, Integer> labels)
	{

		int FilesSize=Files.size();
		HashMap<Integer,String> ProcessingStatus = new HashMap<Integer, String>(); // used for output purposes
		for (int i = 1; i <= 10; i++)
			ProcessingStatus.put((int)(FilesSize * (i / 10d)), i+"0% ("+(int)(FilesSize * (i / 10d))+" out of "+FilesSize+")." );


		ArrayList<Thread> threads = new ArrayList<Thread>();
		for(int i=0;i<NumberOfProcessors;++i){
			threads.add(  (new Thread() {
				int core;
				public void run() {
					try {
						for (int j = 0; j + core <FilesSize; j +=NumberOfProcessors)
						{
							if(debug)
								if (ProcessingStatus.containsKey(j + core))
									System.out.println(dateFormat.format(new Date())+" - Loaded " +ProcessingStatus.get(j + core));
							String file = Files.get(j + core);
							File f = new File(file);
							String fileNameWithOutExt = FilenameUtils.removeExtension(f.getName()); 
							if(j+core==15)
								j=j+1-1;
							analyzeDocumentDemo( Files.get(j+core),labels.get(fileNameWithOutExt) ,core);
						}
					} catch (Exception e) {
						e.printStackTrace();
					} 

				}
				private Thread initialize(int core) {
					this.core = core;
					return this;
				}
			}).initialize(i));
			threads.get(i).start();
		}
		for(int i=0;i<NumberOfProcessors;++i){
			try {
				threads.get(i).join();
			} catch (InterruptedException e) {
				e.printStackTrace();
			} 
		} 
		if(debug)
			System.out.println("Loaded all documents!");

		RemoveVocabTail(); 
		if(debug){
			System.out.println("Vocab size:"+m_Vocabs.size());
			System.out.println("# docs:"+documentsCount);
			System.out.println("# pos:"+TotalPos);
			System.out.println("# neg:"+TotalNeg);
		}

	}
	public void CalculatedNumberOfPositiveAndNegativeDocuments(ArrayList<String>Files,HashMap<String, Integer> labels)
	{

		int FilesSize=Files.size();
		ArrayList<Thread> threads = new ArrayList<Thread>();
		for(int i=0;i<NumberOfProcessors;++i){
			threads.add(  (new Thread() {
				int core;
				public void run() {
					try {
						for (int j = 0; j + core <FilesSize; j +=NumberOfProcessors)
						{
							String file = Files.get(j + core);
							File f = new File(file);
							String fileNameWithOutExt = FilenameUtils.removeExtension(f.getName()); 
							int isPositive=labels.get(fileNameWithOutExt);
							synchronized(lock2) {
								documentsCount++;
								if(isPositive==1)
									TotalPos++;
								else
									TotalNeg++;
							}
						}
					} catch (Exception e) {
						e.printStackTrace();
					} 
				}
				private Thread initialize(int core) {
					this.core = core;
					return this;
				}
			}).initialize(i));
			threads.get(i).start();
		}
		for(int i=0;i<NumberOfProcessors;++i){
			try {
				threads.get(i).join();
			} catch (InterruptedException e) {
				e.printStackTrace();
			} 
		} 
		if(debug){
			System.out.println("Vocab size:"+m_Vocabs.size());
			System.out.println("# docs:"+documentsCount);
			System.out.println("# pos:"+TotalPos);
			System.out.println("# neg:"+TotalNeg);
		}
	}

	public void BuildVectorSpaceModel(ArrayList<String>Files,HashMap<String, Integer> labels,Boolean training){




		int FilesSize=Files.size();
		HashMap<Integer,String> ProcessingStatus = new HashMap<Integer, String>(); // used for output purposes
		for (int i = 1; i <= 10; i++)
			ProcessingStatus.put((int)(FilesSize * (i / 10d)), i+"0% ("+(int)(FilesSize * (i / 10d))+" out of "+FilesSize+")." );


		ArrayList<Thread> threads = new ArrayList<Thread>();
		for(int i=0;i<NumberOfProcessors;++i){
			threads.add(  (new Thread() {
				int core;
				public void run() {
					try {
						for (int j = 0; j + core <FilesSize; j +=NumberOfProcessors)
						{if(debug)
							if (ProcessingStatus.containsKey(j + core))
								System.out.println(dateFormat.format(new Date())+" - Loaded " +ProcessingStatus.get(j + core));
						String file = Files.get(j + core);
						File f = new File(file);
						String fileNameWithOutExt = FilenameUtils.removeExtension(f.getName()); 

						analyzeVSM(Files.get(j+core),labels.get(fileNameWithOutExt) ,documentsCount,core,training);
						}
					} catch (Exception e) {
						e.printStackTrace();

					} 

				}
				private Thread initialize(int core) {
					this.core = core;
					return this;
				}
			}).initialize(i));
			threads.get(i).start();
		}
		for(int i=0;i<NumberOfProcessors;++i){
			try {
				threads.get(i).join();
			} catch (InterruptedException e) {
				e.printStackTrace();
			} 
		}  

	}
	public HashMap<String, Double> FeaturesSelection(FeatureSelectionMethod method,int NumberOfDocumentsInTraining,int NumberOfPostiveDocumentsInTraining, Boolean scale){
		String[] FeaturesSet = m_Vocabs.keySet().toArray(new String[m_Vocabs.size()]);
		HashMap<String, Double> MethodValues=new HashMap<String, Double>();
		if(debug)
			System.out.println(dateFormat.format(new Date())+" - Feature Selection using "+(method==FeatureSelectionMethod.ChiSquare?"ChiSquare":method==FeatureSelectionMethod.InformationGain?"Information Gain":"")+" method.");
		// Calculate corpus-level values (only useful for information gain)
		double py1=NumberOfPostiveDocumentsInTraining/(double)NumberOfDocumentsInTraining;
		double ClassLogSum=-(py1*Math.log(py1)+(1-py1)*Math.log(1-py1));
		for(String finalToken:FeaturesSet)
		{
			double a=m_Vocabs.get(finalToken).getA();
			double b=m_Vocabs.get(finalToken).getB(NumberOfPostiveDocumentsInTraining);
			double c=m_Vocabs.get(finalToken).getC();
			double d=m_Vocabs.get(finalToken).getD(NumberOfDocumentsInTraining-NumberOfPostiveDocumentsInTraining);
			if(method==FeatureSelectionMethod.ChiSquare)// Calculate ChiSquare
				MethodValues.put(finalToken, ((a+b+c+d)*(a*d-b*c)*(a*d-b*c))/ ((a+c)*(b+d)*(a+b)*(c+d)));
			else if(method==FeatureSelectionMethod.InformationGain)	// Calculate InformationGain
			{
				double py1t=a/(a+c);
				double py1tbar=b/(b+d);
				double pt=(a+c)/(a+b+c+d);
				MethodValues.put(finalToken, ClassLogSum+
						pt*((py1t==0?0:py1t*Math.log(py1t))+(py1t==1?0:(1-py1t)*Math.log(1-py1t)))
						+(1-pt)*((py1tbar==0?0:py1tbar*Math.log(py1tbar))+(py1tbar==1?0:(1-py1tbar)*Math.log(1-py1tbar))));
			}
		} 
		// scale to 0-1 range
		if(scale){
			// get max and min values
			Double min = Collections.min(MethodValues.values());
			Double max = Collections.max(MethodValues.values());

			for(String key:MethodValues.keySet()){
				MethodValues.put(key, (MethodValues.get(key)-min)/(max-min));
			}
		}
		return MethodValues;
	}
	public ArrayList<String> FeaturesSelection(FeatureSelectionMethod method, int NumberOfFeaturesToKeep,Boolean Save,int NumberOfDocumentsInTraining,int NumberOfPostiveDocumentsInTraining){
		ArrayList<String> FeaturesToKeep=new ArrayList<String>();
		String[] FeaturesSet = m_Vocabs.keySet().toArray(new String[m_Vocabs.size()]);
		HashMap<String, Double> MethodValues=new HashMap<String, Double>();
		if(debug)
			System.out.println(dateFormat.format(new Date())+" - Feature Selection using "+(method==FeatureSelectionMethod.ChiSquare?"ChiSquare":method==FeatureSelectionMethod.InformationGain?"Information Gain":"")+" method.");
		// Calculate corpus-level values (only useful for information gain)
		double py1=NumberOfPostiveDocumentsInTraining/(double)NumberOfDocumentsInTraining;
		double ClassLogSum=-(py1*Math.log(py1)+(1-py1)*Math.log(1-py1));
		for(String finalToken:FeaturesSet)
		{
			double a=m_Vocabs.get(finalToken).getA();
			double b=m_Vocabs.get(finalToken).getB(NumberOfPostiveDocumentsInTraining);
			double c=m_Vocabs.get(finalToken).getC();
			double d=m_Vocabs.get(finalToken).getD(NumberOfDocumentsInTraining-NumberOfPostiveDocumentsInTraining);
			if(method==FeatureSelectionMethod.ChiSquare)// Calculate ChiSquare
				MethodValues.put(finalToken, ((a+b+c+d)*(a*d-b*c)*(a*d-b*c))/ ((a+c)*(b+d)*(a+b)*(c+d)));
			else if(method==FeatureSelectionMethod.InformationGain)	// Calculate InformationGain
			{
				double py1t=a/(a+c);
				double py1tbar=b/(b+d);
				double pt=(a+c)/(a+b+c+d);
				MethodValues.put(finalToken, ClassLogSum+
						pt*((py1t==0?0:py1t*Math.log(py1t))+(py1t==1?0:(1-py1t)*Math.log(1-py1t)))
						+(1-pt)*((py1tbar==0?0:py1tbar*Math.log(py1tbar))+(py1tbar==1?0:(1-py1tbar)*Math.log(1-py1tbar))));
			}
		}
		// Sort
		List<Map.Entry<String,Double>> entries = new LinkedList<Map.Entry<String,Double>>(MethodValues.entrySet());
		Collections.sort(entries, new Comparator<Map.Entry<String,Double>>() {
			@Override
			public int compare(Entry<String,Double> o1, Entry<String,Double> o2) {
				return o2.getValue().compareTo(o1.getValue());
			}
		});
		Map<String,Double> sortedMap = new LinkedHashMap<String,Double>();
		for(Map.Entry<String,Double> entry: entries){
			sortedMap.put(entry.getKey(), entry.getValue());
		}
		if(Save){
			// Save to csv file
			try {
				FileWriter fstream = new FileWriter("./featureSelection"+(method==FeatureSelectionMethod.ChiSquare?"Chi":method==FeatureSelectionMethod.InformationGain?"IG":"")+".csv", false);
				BufferedWriter out = new BufferedWriter(fstream);

				for(Map.Entry<String,Double> entry: sortedMap.entrySet())
					out.write(entry.getKey()+","+entry.getValue() +"\n");
				out.close();
				if(debug)
					System.out.println("featureSelection "+(method==FeatureSelectionMethod.ChiSquare?"Chi":method==FeatureSelectionMethod.InformationGain?"IG":"")+" Saved!");

			} catch (Exception e) {
				e.printStackTrace(); 
			}
		}
		// remove unwanted features
		Iterator<String> iterator = sortedMap.keySet().iterator();
		int index=0;
		while (iterator.hasNext()&&index++<NumberOfFeaturesToKeep) {
			String feature=iterator.next();
			if(method==FeatureSelectionMethod.InformationGain||(method==FeatureSelectionMethod.ChiSquare&&sortedMap.get(feature)>3.841)) // filter out insignificant terms for chiSquare
				FeaturesToKeep.add(feature);
		}
		return FeaturesToKeep;
	}

	public void ConstructFeatures(int NumberOfFeatures){
		// select features 
		ArrayList<String>FeaturesToKeepUsingChi= FeaturesSelection(FeatureSelectionMethod.ChiSquare,NumberOfFeatures,false,documentsCount,TotalPos);
		ArrayList<String>FeaturesToKeepUsingIG= FeaturesSelection(FeatureSelectionMethod.InformationGain,NumberOfFeatures,false,documentsCount,TotalPos);
		// Remove from the original Vocabulary
		Set<String> set = m_Vocabs.keySet();
		Iterator<String> itr = set.iterator();
		while (itr.hasNext())
		{
			String key = itr.next();
			if (!FeaturesToKeepUsingChi.contains(key)||!FeaturesToKeepUsingIG.contains(key))  
				itr.remove();
		}
		// re-adjust ID to use it as index for classifier parameters
		String[] words = m_Vocabs.keySet().toArray(new String[m_Vocabs.size()]);
		for (int i=1;i<=m_Vocabs.size();++i)
			m_Vocabs.get(words[i-1]).setID(i);
		if(debug)
			System.out.println("Selected "+m_Vocabs.size() +" features.");
	}
	public LogisticRegressionClassifier BuildClassifier(int NumberOfProcessors){

		// Build Training and Testing Sets
		if(debug)
			System.out.println(dateFormat.format(new Date())+" Building Training and testing sets:");
		double[]TrainingTrueLabels=new double[trainingDocuments.size()];
		// get labels
		for(int i=0;i<trainingDocuments.size();++i) 
			TrainingTrueLabels[i]=trainingDocuments.get(i).getLabel();
		if(debug)
			System.out.println(dateFormat.format(new Date())+" Training Classifier:");
		// Create Classifier
		LogisticRegressionClassifier Classifier=new LogisticRegressionClassifier(m_Vocabs.size()+1, NumberOfProcessors , m_Vocabs );
		// train
		Classifier.Train(trainingDocuments, TrainingTrueLabels);
		return Classifier;
	}
	public double[] EvaluateClassifier(LogisticRegressionClassifier Classifier){
		double[] performance=new double[13];
		if(debug)
			System.out.println(dateFormat.format(new Date())+" Test Classifier:");
		// Test Classifier
		int CorrectClassifications=0;
		int PosClassified=0,PosCorrectClassified=0;
		int NegClassified=0,NegCorrectClassified=0; 
		int totalPos=0,totalNeg=0;
		for(int i=0;i<testingDocuments.size();++i){
			Document testingDoc=testingDocuments.get(i);
			double testingLabel=testingDoc.getLabel();
			double label=Classifier.Classify(testingDoc ,0.5);
			if(label==testingLabel)
				CorrectClassifications++;
			if(label==1.0d)PosClassified++;else NegClassified++;
			if(label==1.0d&label==testingLabel)PosCorrectClassified++; 
			if(label==0d&label==testingLabel)NegCorrectClassified++; 
			if(testingLabel==1d) totalPos++; else totalNeg++;
		}


		double posPre=PosCorrectClassified/(double)PosClassified;
		double posRec=PosCorrectClassified/(double)totalPos;
		double posF=2*posPre*posRec/(posPre+posRec);
		double negPre=NegCorrectClassified/(double)NegClassified;
		double negRec=NegCorrectClassified/(double)totalNeg;
		double negF=2*negPre*negRec/(negPre+negRec);

		performance[0]=CorrectClassifications/(double)testingDocuments.size(); 
		performance[1]=posPre;
		performance[2]=posRec;
		performance[3]=posF;
		performance[4]=negPre;
		performance[5]=negRec;
		performance[6]=negF;
		performance[7]=NegClassified;
		performance[8]=NegCorrectClassified;
		performance[9]=totalNeg;
		performance[10]=PosClassified;
		performance[11]=PosCorrectClassified; 
		performance[12]=totalPos; 

		return performance;
	}
	public void LoadVocab(String filename)
	{
		try {
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
			String line;

			while ((line = reader.readLine()) != null) {
				//it is very important that you perform the same processing operation to the loaded stopwords
				//otherwise it won't be matched in the text content
				if (line.isEmpty())continue;

				String[] values=line.split(",");
				m_Vocabs.put(values[1],new Token(Integer.parseInt(values[0]),values[1],Double.parseDouble(values[2]),Double.parseDouble(values[3]),Double.parseDouble(values[4])));

			}
			reader.close();

		} catch(IOException e){
			System.err.format("[Error]Failed to open file !!" );
		}
	}
	 
	 
	 public void RunAnalyzer(ExperimentalSetup setup, String DataDirPath,String labelsFilePath,String groupsLabelsFilePath,String cv_documents,String logFileName ) {	
		// check if log file exist. If so, delete it
		File f=new File(logFileName+".txt");
		if(f.exists())
			f.delete();
		// Load Files and Labels
		ArrayList<String>Files= GetFiles(DataDirPath, ".txt");
		HashMap<String, Integer> labels = GetLabels(labelsFilePath);
		HashMap<String, Integer> groupLabels = GetLabels(groupsLabelsFilePath);
		String[] groups=groupLabels.keySet().toArray(new String[groupLabels.size()]);
		int numberOfViolentGroups=0,numberOfNonViolentGroups=0;
		// CV or LOGO-CV
		//		- Build Selected Vocabulary from training
		//		- Build Vector Space Models (Only IDF)
		//		- Train a Logistic Regression Classifier
		//		- Test
		ArrayList<String> screenOutput=new ArrayList<String>();
		ArrayList<ArrayList<String>> testingCVDocs=new ArrayList<ArrayList<String>>();
		if(setup==ExperimentalSetup.CV){
			// load cv documents
			testingCVDocs=LoadCVTestingDocuments(cv_documents);
		}
		else if(setup==ExperimentalSetup.LOGOCV){
			// get groups 
			for(String group:groups){
				ArrayList<String> testingDocs=new ArrayList<String>();
				for(String file : Files){
					String fileName=new File(file).getName();
					if(fileName.startsWith(group))
						testingDocs.add(fileName);
				}
				testingCVDocs.add(testingDocs);
				if(groupLabels.get(group)==1)
					numberOfViolentGroups++;
				else
					numberOfNonViolentGroups++;
			}
		}
		int numberOfRuns=testingCVDocs.size();
		double[] totalPerformance=new double[13];


		for(int i=0;i<numberOfRuns;i++){
			// get training and testing docs
			ArrayList<String> trainingFiles=new ArrayList<String>();
			ArrayList<String> testingFiles=new ArrayList<String>();
			for(String file : Files){

				if(!testingCVDocs.get(i).contains(new File(file).getName()))
					trainingFiles.add(file);
				else
					testingFiles.add(file);
			}
			// init
			init();
			String vocabFile="";
			if(setup==ExperimentalSetup.CV)
				vocabFile="output\\f_"+i+"_vocab";
			else if(setup==ExperimentalSetup.LOGOCV)
				vocabFile= "output\\g_"+groups[i]+"_vocab";


			if(new File(vocabFile+".csv").exists())
			{
				// load selected Vocab
				LoadVocab(vocabFile+".csv");
				// calculate number of positive and negative documents in training so we can compute IDF
				CalculatedNumberOfPositiveAndNegativeDocuments(trainingFiles, labels);
			}
			else
			{
				// build vocab
				BuildVocab(trainingFiles, labels);
				// select features
				ConstructFeatures(10000);
				Save(m_Vocabs, vocabFile);
			}



			// train or load classification model
			LogisticRegressionClassifier Classifier=null;
			String topFeasturesFile="";
			String classifierFile="";
			if(setup==ExperimentalSetup.CV)
			{
				topFeasturesFile="output\\f_"+i+"_topFeatures";
				classifierFile="output\\f_"+i;
			}
			else if(setup==ExperimentalSetup.LOGOCV)
			{
				topFeasturesFile="output\\g_"+groups[i]+"_topFeatures";
				classifierFile="output\\g_"+groups[i];
			} 
			// build VSM for testing files
			BuildVectorSpaceModel(testingFiles, labels, false);
			if(testingDocuments.size()==0)
				continue;
			if(new File(classifierFile+".classifer").exists())
			{
				// Load Classifier
				Classifier=new LogisticRegressionClassifier(m_Vocabs.size()+1, NumberOfProcessors , m_Vocabs);
				Classifier.Load(classifierFile);
			}
			else
			{
				// build VSM for training files
				BuildVectorSpaceModel(trainingFiles, labels, true);

				// Train and save classifier
				Classifier=BuildClassifier(NumberOfProcessors);
				Classifier.Save(classifierFile);
				Classifier.saveTopFeatures(topFeasturesFile);
			}


			// Evaluate
			double[] performance=EvaluateClassifier(Classifier);
			if(setup==ExperimentalSetup.CV)
				screenOutput.add("Fold "+i+", Accuracy: "+performance[0]); 
			else if(setup==ExperimentalSetup.LOGOCV)
				screenOutput.add("Group "+groups[i]+", Accuracy: "+performance[0]); 
			screenOutput.add("0 class ("+performance[7]+"/"+performance[8]+"/"+performance[9]+"): Precision="+performance[4]+",Recall="+performance[5]+",F1-Measure="+performance[6]);
			screenOutput.add("1 class ("+performance[10]+"/"+performance[11]+"/"+performance[12]+"): Precision="+performance[1]+",Recall="+performance[2]+",F1-Measure="+performance[3]);
			try {
				FileWriter fstream = new FileWriter(logFileName+".txt", true);
				BufferedWriter out = new BufferedWriter(fstream);
				int outputIndex=screenOutput.size()-3;
				out.write(screenOutput.get(outputIndex++)+"\n");
				out.write(screenOutput.get(outputIndex++)+"\n");
				out.write(screenOutput.get(outputIndex)+"\n");
				out.close();
				//System.out.println(FileName+" Classifier Saved!");
			} catch (Exception e) {
				e.printStackTrace(); 
			} 
			for(int k=0;k<13;++k)
				totalPerformance[k]+=performance[k];
		}
		for(int k=0;k<screenOutput.size();++k)
			System.out.println(screenOutput.get(k));
		// average performance
		System.out.println("Avg Accuracy: "+totalPerformance[0]/(double)numberOfRuns);
		if(setup==ExperimentalSetup.CV){
			System.out.println("0 class ("+totalPerformance[7]/(double)numberOfRuns+"/"+totalPerformance[8]/(double)numberOfRuns+"/"+totalPerformance[9]/(double)numberOfRuns+"): Precision="+totalPerformance[4]/(double)numberOfRuns+",Recall="+totalPerformance[5]/(double)numberOfRuns+",F1-Measure="+totalPerformance[6]/(double)numberOfRuns);
			System.out.println("1 class ("+totalPerformance[10]/(double)numberOfRuns+"/"+totalPerformance[11]/(double)numberOfRuns+"/"+totalPerformance[12]/(double)numberOfRuns+"): Precision="+totalPerformance[1]/(double)numberOfRuns+",Recall="+totalPerformance[2]/(double)numberOfRuns+",F1-Measure="+totalPerformance[3]/(double)numberOfRuns);
		}
		else if(setup==ExperimentalSetup.LOGOCV)
		{
			System.out.println("0 class ("+totalPerformance[7]/(double)numberOfNonViolentGroups+"/"+totalPerformance[8]/(double)numberOfNonViolentGroups+"/"+totalPerformance[9]/(double)numberOfNonViolentGroups+"): Precision="+totalPerformance[4]/(double)numberOfNonViolentGroups+",Recall="+totalPerformance[5]/(double)numberOfNonViolentGroups+",F1-Measure="+totalPerformance[6]/(double)numberOfNonViolentGroups);
			System.out.println("1 class ("+totalPerformance[10]/(double)numberOfViolentGroups+"/"+totalPerformance[11]/(double)numberOfViolentGroups+"/"+totalPerformance[12]/(double)numberOfViolentGroups+"): Precision="+totalPerformance[1]/(double)numberOfViolentGroups+",Recall="+totalPerformance[2]/(double)numberOfViolentGroups+",F1-Measure="+totalPerformance[3]/(double)numberOfViolentGroups);

		}

	}
	 public void RunAnalyzerOnOneGroup( String DataDirPath,String labelsFilePath,String[] groupsToRemove,String targetGroup ) {	

			// Load Files and Labels
			ArrayList<String>Files= GetFiles(DataDirPath, ".txt");
			HashMap<String, Integer> labels = GetLabels(labelsFilePath);
			// get training and testing docs
			ArrayList<String> trainingFiles=new ArrayList<String>();
			ArrayList<String> testingFiles=new ArrayList<String>();



			for(String file : Files){
				String fileName=new File(file).getName();
				if(fileName.startsWith(targetGroup))
					testingFiles.add(file);
				else
				{
					Boolean remove=false;
					for(String groupToRemove:groupsToRemove)
						if(fileName.startsWith(groupToRemove)){
							remove=true;break;
						}
					if(!remove)
						trainingFiles.add(file);
				}
			}
			// init
			init();

			// build vocab
			BuildVocab(trainingFiles, labels);
			// select features
			ConstructFeatures(10000);


			// train or load classification model
			LogisticRegressionClassifier Classifier=null; 
			
			// build VSM for testing files
			BuildVectorSpaceModel(testingFiles, labels, false);
			if(testingDocuments.size()==0)
				return;

			// build VSM for training files
			BuildVectorSpaceModel(trainingFiles, labels, true);

			// Train and save classifier
			Classifier=BuildClassifier(NumberOfProcessors);
			Classifier.saveTopFeatures("topfeatures.csv");
			Save(m_Vocabs, "vocab.csv");


			// Evaluate
			double[] performance=EvaluateClassifier(Classifier);

			System.out.println("Group "+targetGroup+", Accuracy: "+performance[0]); 
			System.out.println("0 class ("+performance[7]+"/"+performance[8]+"/"+performance[9]+"): Precision="+performance[4]+",Recall="+performance[5]+",F1-Measure="+performance[6]);
			System.out.println("1 class ("+performance[10]+"/"+performance[11]+"/"+performance[12]+"): Precision="+performance[1]+",Recall="+performance[2]+",F1-Measure="+performance[3]);


		}
}
