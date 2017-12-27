/**
 * 
 */
package structures;

 
import java.util.HashMap;
import java.util.Iterator;
import java.util.Set;



public class Document {

public String UniqueID;
	public HashMap<String,Double> m_VSM;
	private double Norm;
	private String fileName;
	public String getFileName() {
		return fileName;
	}

	public void setFileName(String fileName) {
		this.fileName=fileName;
	}
	public void clearContent(){
		Content=null;
	}

	private String Content;
	public void CalculateNorm()
	{
		Norm=0;
		// Calculate the d_j norm
		Set<String> 	set = m_VSM.keySet();
		Iterator<String>  itr = set.iterator();
		while (itr.hasNext())
		{
			String key = itr.next();
			Norm+=m_VSM.get(key)*m_VSM.get(key);
		}
	}
	public boolean isEmpty() {
		return Content==null || Content.isEmpty();
	}


	public Document( ) {
		m_VSM=new HashMap<String, Double>();
	}
	public double getValueFromVSM(String key)
	{
		//return m_VSM.containsKey(key)?m_VSM.get(key):0;
		return m_VSM.get(key);
	}
	public String getContent() {
		return Content;
	}

	public void setContent(String content) {
		Content = content;
	}
	public double getLabel()
	{
		return isViolent;
	}
	 

	


	public void setLabel(double isViolent) {
		this.isViolent = isViolent;
	}

	public double getNorm() {
		return Norm;
	}
 
	private double isViolent;
	
}
