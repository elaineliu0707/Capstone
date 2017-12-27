package analyzer;

import java.util.ArrayList;
import java.util.HashMap; 
import java.util.Random; 

import edu.cmu.cs.bungee.lbfgs.LBFGS;
import structures.Document;
import structures.Token;

public class LogisticRegressionClassifier extends Classifier {

	 
	
	
	public LogisticRegressionClassifier(int NumberOfParameters,int NumberOfProcessors ,HashMap<String, Token> vocabulary ) {
		super(NumberOfParameters,NumberOfProcessors ,vocabulary );
		
 
	}
	public void Init(){
		for(int i=0;i<NumberOfParameters;++i)Parameters[i]=0;
	}
	public double getValue() {
		double[] value =new double[NumberOfProcessors];
		ArrayList<Thread> threads = new ArrayList<Thread>();
		for(int i=0;i<NumberOfProcessors;++i){
			threads.add(  (new Thread() {
				int core;
				public void run() {
					try {
						 
						for(int j=0;j+ core<trainingSize;j +=NumberOfProcessors){
							double h_x=ClassifiedValues[j+ core];
							value[core]+=TrueLabels[j+ core]==1?(h_x==0?-1.0e4:Math.log(h_x))
									:(h_x==1?-1.0e4:Math.log(1-h_x));
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
		double ValueFinal=0;
		for(int i=0;i<NumberOfProcessors;++i){
			try {
				threads.get(i).join();
				ValueFinal+=value[i];
			} catch (InterruptedException e) {
				e.printStackTrace();
			} 
		} 
		
	
	 
	 
		return -1*ValueFinal +(L2Norm()/2d);
	}
	public double L2Norm()
	{
		double value=0;
		for(int i=0;i<NumberOfParameters;++i) 
			value+=(Parameters[i] )*(Parameters[i] );
		return  value ;
	}
	 public void AddNoise(double StandardDeviation){
		 Random Rgen=new Random();
		for(int i=0;i<NumberOfParameters;++i) 
			Parameters[i]+=Rgen.nextGaussian()*StandardDeviation;
	 }
	 public Object locker=new Object();
	public void getValueGradient(double[] gradient) {
		// Calculate error
		double[] errors=new double[trainingSize];
		ArrayList<Thread> threads = new ArrayList<Thread>();
			for(int i=0;i<NumberOfProcessors;++i){
				threads.add(  (new Thread() {
					int core;
					public void run() {
						try {
							for(int j=0;j+ core< trainingSize;j +=NumberOfProcessors)
								errors[j+ core]=TrueLabels[j+ core]-ClassifiedValues[j+ core] ;
							 
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
			for(int j=0;j< trainingSize;j++)
				errors[j]=TrueLabels[j]-ClassifiedValues[j] ;
			
			// Update parameters
			double[] updates=new double[NumberOfParameters];
			threads = new ArrayList<Thread>();
			for(int i=0;i<NumberOfProcessors;++i){
				threads.add(  (new Thread() {
					int core;
					public void run() {
						try {
							
							for(int k=0;k+ core< trainingSize;k +=NumberOfProcessors){
								synchronized(locker) {
									updates[0]+=errors[k+core]; // bais
								}
								Document currentDoc=TrainingSet.get(k+core);
								for(String word : currentDoc.m_VSM.keySet())
									synchronized(locker) {
										updates[vocabulary.get(word).getID()]+=errors[k+core]*currentDoc.getValueFromVSM(word);
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
			 for (int j = 0; j  <NumberOfParameters; j ++)
				 gradient[j]=-1*updates[j] +Parameters[j]  ;
	}
	public int getNumParameters() { return NumberOfParameters; }
	public double getParameter(int i) { return Parameters[i]; }
	public void getParameters(double[] buffer) {
		for(int j=0;j<NumberOfParameters;++j) 
			buffer[j] = Parameters[j];
	}
	public void setParameter(int i, double r) {
		Parameters[i] = r;
	}
	public void setParameters(double[] newParameters) {
		for(int j=0;j<NumberOfParameters;++j) 
			Parameters[j] = newParameters[j];  
	}
	 
	@Override
	public double Classify(Document NewInstance) {
		double value=Parameters[0]; // bais
		for(String word : NewInstance.m_VSM.keySet())
			value+=Parameters[vocabulary.get(word).getID()]*NewInstance.getValueFromVSM(word);
		return 1/(double)(1+Math.exp(-1*value)); // sigmod function
	}

	@Override
	public double Classify(Document NewInstance, double Threshold) {
		return Classify(NewInstance)>Threshold?1:0;
	}
	 
	ArrayList<Document> TrainingSet;
	private double[] ClassifiedValues;
	double[] TrueLabels;
	 int trainingSize;
	@Override
	public void Train(ArrayList<Document> TrainingSet,double[] TrueLabels) {
		this.TrainingSet=TrainingSet;
		this.TrueLabels=TrueLabels;
		trainingSize=TrainingSet.size();
		// Minimize the cost function using LBFGS
 	 
 		int ndim = NumberOfParameters;
		double []  gradient,diag;
		//Initialize the parameters to lbfgs 
	 
		gradient = new double [ ndim ];
		diag = new double [ ndim ];


		double f, eps, xtol ;
		int iprint [ ] , iflag[] = new int[1], icall, m;
		iprint = new int [ 2 ];
		boolean diagco;
		m=6;
		iprint [0] = -1;
		iprint [1] = 0;
		diagco= false;
		eps= 1.0e-4;
		xtol= 1.0e-3;
		icall=0;
		iflag[0]=0;

		
	
		do
		{
			// Classify 
 			ClassifiedValues=new double[trainingSize];
 		 
 			ArrayList<Thread> threads = new ArrayList<Thread>();
 			for(int i=0;i<NumberOfProcessors;++i){
 				threads.add(  (new Thread() {
 					int core;
 					public void run() {
 						try {
 							for (int j = 0; j + core <trainingSize; j +=NumberOfProcessors){
 								ClassifiedValues[j+ core]=Classify(TrainingSet.get(j+ core));
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
			 getValueGradient(gradient); 
			 
			f= getValue();
			try
			{
				LBFGS.lbfgs (ndim, m , Parameters, f , gradient , diagco , diag , iprint , eps , xtol , iflag );
//				double mang=0;
//				for(int i=0;i<gradient.length;++i)
//					mang+=gradient[i]*gradient[i];
//				 
				//System.out.println(icall);
			 // System.out.println(icall+" "+f+" "+mang  );
				 
			}
			catch (LBFGS.ExceptionWithIflag e)
			{
				System.err.println( "Sdrive: lbfgs failed.\n"+e );
				return;
			}
			 
			icall += 1;

		}
		while ( iflag[0] != 0 && icall <= 400 );
//		double mang=0;
//		for(int i=0;i<gradient.length;++i)
//			mang+=gradient[i]*gradient[i];
		 
		 
	  // System.out.println(icall+" "+f+" "+mang  );
	}


 

}
