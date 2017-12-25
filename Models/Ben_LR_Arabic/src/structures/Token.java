/**
 * 
 */
package structures;
 


public class Token {

 
	int m_id; // the numerical ID you assigned to this token/N-gram
	public int getID() {
		return m_id;
	}

	public void setID(int id) {
		this.m_id = id;
	}

	String m_token; // the actual text content of this token/N-gram
	public String getToken() {
		return m_token;
	}

	public void setToken(String token) {
		this.m_token = token;
	}

	double m_value; // frequency or probability of this token/N-gram
	public double getValue() {
		return m_value;
	}

	public void setValue(double value) {
		this.m_value =value;
	}	
	double m_PosDF; // frequency or probability of this token/N-gram
	public double getPosDF() {
		return m_PosDF;
	}

	public void setPosDF(double PosDF) {
		this.m_PosDF =PosDF;
	}	
	double m_NegDF; // frequency or probability of this token/N-gram
	public double getNegDF() {
		return m_NegDF;
	}
	public double getDF() {
		return m_NegDF+m_PosDF;
	}
	public double getA() {
		return m_PosDF;
	}
	public double getB(int NumberOfPositiveDocsInTraining) {
		return NumberOfPositiveDocsInTraining-m_PosDF;
	}
	public double getC() {
		return m_NegDF;
	}
	public double getD(int NumberOfNegativeDocsInTraining) {
		return NumberOfNegativeDocsInTraining-m_NegDF;
	}
	public void setNegDF(double NegDF) {
		this.m_NegDF =NegDF;
	}	
	@Override
    public String toString() {
        return String.format( m_value+"");
    }
	//default constructor
	public Token(String token) {
		m_token = token;
		m_id = -1;
		m_value = 0;		
	}
	public Token(Token token) {
		m_token = token.getToken();
		m_id = token.getID() ;
		m_value = token.getValue() ;		
	}
	//default constructor
	public Token(int id, String token) {
		m_token = token;
		m_id = id;
		m_value = 0;
		m_NegDF=0;
		m_PosDF=0;
	}
	//default constructor
		public Token(int id, String token,double value,double PosDF,double NegDF) {
			m_token = token;
			m_id = id;
			m_value = value;		
			m_PosDF=PosDF;
			m_NegDF=NegDF;
		}
		public Token(int id, String token,double value ) {
			m_token = token;
			m_id = id;
			m_value = value;		
	 
		}
}
