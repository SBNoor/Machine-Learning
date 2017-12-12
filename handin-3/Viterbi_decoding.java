/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package viterbi_decoding;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import static java.lang.Math.log;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;


public class Viterbi_decoding {
    final static String DESCRIPTION = "Program requires 3 parameters and can be run as follows:\n" +
"   java -jar viterbi_decoding.jar [FILE1] [FILE2] [FILE3]\n\n" +
" EX.\n" +
"    java -jar viterbi_decoding.jar hmm-tm.txt sequences-project2.txt sequences-project2-viterbi-output.txt\n\n" +
"   [FILE1] - file with HMM parameters\n" +
"   [FILE2] - input file with sequences in fasta format\n" +
"   [FILE3] - name of an output file.\n\n" +

"";
    HashMap<String, Integer> hidden = new HashMap<>();
    HashMap<String, Integer> observables = new HashMap<>();
    char[] hidd;
    double[] pi;
    double[][] transitions;
    double[][] emissions;
//    char[] X = "MVSAKKVPAIAMSFGVSFALLHFLCLAACLNESPGQNQKEEKLCPENFTRILDSLLDGYDNRLRPGFGGPVTEVKTDIYVTSFGPVSDVEMEYTMDVFFRQTWIDKRLKYDGPIEILRLNNMMVTKVWTPDTFFRNGKKSVSHNMTAPNKLFRIMRNGTILYTMRLTISAECPMRLVDFPMDGHACPLKFGSYAYPKSEMIYTWTKGPEKSVEVPKESSSLVQYDLIGQTVSSETIKSITGEYIVMTVYFHLRRKMGYFMIQTYIPCIMTVILSQVSFWINKESVPARTVFGITTVLTMTTLSISARPISLPKVSYATAMDWFIAVCFAFVFSALIEFAAVNYFTNVQMEKAKRKTSKAPQEISAAPVLREKHPETPLQNTNANLSMRKRANALVHSESDVGSRTDVGNHSSKSSTVVQGSSEATPQSYLASSPNPFSRANAAETISAARAIPSALPSTPSRTGYVPRQVPVGSASTQHVFGSRLQRIKTTVNSIGTSGKLSATTTPSAPPPSGSGTSKIDKYARILFPVTFGAFNMVYWVVYLSKDTMEKSESLM".toCharArray();
    char[] X;
    ArrayList<String> inSeq = new ArrayList<>();
    ArrayList<String> seqNames = new ArrayList<>();
    ArrayList<String> Zlist = new ArrayList<>();
    double[] logprob;
    double[] logJP;
    char[] Z;
    double[][] omega;
    int K;
    int N;
    //String[] param = {"../hmm-tm.txt", "../sequences-project2.txt", "../sequences-project2out.txt"};
    String[] param;
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        // TODO code application logic here
        if(args.length!=3){
            System.out.println(DESCRIPTION);
            System.exit(0);
        }
        Viterbi_decoding vd = new Viterbi_decoding();
        vd.Viterbi_decoding(args);
    }
    
    public void Viterbi_decoding(String[] args){
        param=args;
        readMM(param[0]);
        System.out.println("Reading input...");
        readInp(param[1]);
        System.out.println("Reading DONE!!!");
        System.out.println("Sequence length: " + inSeq.get(0).length());
        
        K = pi.length;
        logprob = new double[inSeq.size()];
        logJP = new double[inSeq.size()];
        for(int m=0;m<inSeq.size();m++){
            X = inSeq.get(m).toCharArray();
            N = X.length;
            omega = new double[K][N];
            for(int k=0;k<K;k++){
                omega[k][0] = log(pi[k]) + log(emissions[k][observables.get(String.valueOf(X[0]))]);
            }

            for(int n=1;n<N;n++){
                for(int k=0;k<K;k++){
                    omega[k][n]=log(emissions[k][observables.get(String.valueOf(X[n]))]) + getMaxOmega(n-1, k);
                }
            }
            logprob[m] = getJP();

            //backtrack
            Z = new char[N];
            Z[N-1] = hidd[getMaxOmegaInd(N-1)];
            for(int n=N-2;n>=0;n--){
                Z[n] = hidd[getIndMax(n)];
            }
            Zlist.add(String.valueOf(Z));
            logJP[m] = calcLogJointProb(X,Z);
        }
        System.out.println("; Viterbi-decodings of "+param[1]+" using HMM "+param[0]);
        System.out.println("");
        /*
        for(int i=0;i<inSeq.size();i++){
            System.out.println(seqNames.get(i));
            System.out.println(inSeq.get(i));
            System.out.println("#");
            System.out.println(Zlist.get(i));
            System.out.println("; log P(x,z) = "+logprob[i]);
            System.out.println("");
        }
*/
        saveOut(param[2]);
   }
    
    //Calc log-joint-prob with given observables and hidden states
    private double calcLogJointProb(char[] X, char[] Z){
        double p = 0;
        p=log(pi[hidden.get(String.valueOf(Z[0]))]) + log(emissions[hidden.get(String.valueOf(Z[0]))][observables.get(String.valueOf(X[0]))]);
        for(int i=1; i<X.length;i++){
            p=p + log(transitions[hidden.get(String.valueOf(Z[i-1]))][hidden.get(String.valueOf(Z[i]))]) + log(emissions[hidden.get(String.valueOf(Z[i]))][observables.get(String.valueOf(X[i]))]);
        }
        
        return p;
    }
    
    private int getIndMax(int n){
        double max = Double.NEGATIVE_INFINITY;
        int ind = -1;
        for(int k=0;k<K;k++){
            double val = log(emissions[hidden.get(String.valueOf(Z[n+1]))][observables.get(String.valueOf(X[n+1]))])+omega[k][n]+log(transitions[k][hidden.get(String.valueOf(Z[n+1]))]);
            if(max<val){
                max=val;
                ind=k;
            }
        }
        return ind;      
    }
    
    private double getMaxOmega(int n,int k){
        double res = Double.NEGATIVE_INFINITY;
        for(int j=0;j<K;j++){
            double dob = omega[j][n] + log(transitions[j][k]);
            if(res<dob){
                res=dob;
            }
        }
        return res;
    }
    
    private double getJP(){
        double res = Double.NEGATIVE_INFINITY;
        for(int k=0;k<K;k++){
            if(res<omega[k][N-1]){
                res=omega[k][N-1];
            }
        }
        return res;
    }

    private int getMaxOmegaInd(int n){
        double res = Double.NEGATIVE_INFINITY;
        int ind = -1;
        for(int k=0;k<K;k++){
            if(res<omega[k][n]){
                res=omega[k][n];
                ind=k;
            }
        }
        return ind;
    }
    
    private void readMM(String input){
        try{
            BufferedReader br1 = new BufferedReader(new FileReader(input));
            String line = br1.readLine();
            while(line!=null){
                if(line.contains("hidden")){
                    String[] l = br1.readLine().split(" ");
                    hidd = new char[l.length];
                    for(int i=0;i<l.length;i++){
                        hidden.put(l[i], i);
                        hidd[i] = l[i].charAt(0);
                    }
                }
                if(line.contains("observables")){
                    String[] l = br1.readLine().split(" ");
                    for(int i=0;i<l.length;i++){
                        observables.put(l[i], i);
                    }
                }
                if(line.contains("pi")){
                    String[] l = br1.readLine().split(" ");
                    pi = new double[l.length];
                    for(int i=0;i<pi.length;i++){
                        pi[i] = Double.valueOf(l[i]);
                    }
                }
                if(line.contains("transitions")){
                    String[] l = br1.readLine().split(" ");
                    int n = l.length;
                    transitions = new double[n][n];
                    for(int i=0;i<n;i++){
                        for(int j=0;j<n;j++){
                            transitions[i][j] = Double.valueOf(l[j]);
                        }
                        l = br1.readLine().split(" ");
                    }
                }
                if(line.contains("emissions")){
                    String[] l = br1.readLine().split(" ");
                    int n = l.length;
                    emissions = new double[pi.length][n];
                    for(int i=0;i<pi.length;i++){
                        for(int j=0;j<n;j++){
                            emissions[i][j] = Double.valueOf(l[j]);
                        }
                        l = br1.readLine().split(" ");
                    }
                }
                line = br1.readLine();
            }
        }catch(Exception e){
            e.printStackTrace();
        }                
    }
    
    private void readInp(String input){
        String S=null;
        try{
        S = new String (Files.readAllBytes(Paths.get(input)),Charset.forName("UTF-8"));
            seqNames.add(S.substring(S.indexOf(">"), S.indexOf("\n",S.indexOf(">"))));
            S=S.substring(S.indexOf("\n",S.indexOf(">")));            
            S=S.replace("\r", "");
            S=S.replace("\n", "");
            inSeq.add(S);
        }catch(Exception e){
            e.printStackTrace();
        }

//        try{
//            BufferedReader br1 = new BufferedReader(new FileReader(input));
//            String line = br1.readLine();
//            String seq;
//            while(line!=null){
//                if(line.startsWith(">")){
//                    seqNames.add(line);
//                    line = br1.readLine();
//                    while(line.isEmpty()){
//                        line = br1.readLine();
//                    }
//                    seq = line.trim();
//                    line = br1.readLine();
//                    while(line!=null){
//                        seq+=line.trim();
//                        line = br1.readLine();
//                    }
//                    inSeq.add(seq);
//                }
//
//                line = br1.readLine();
//            }
//        }catch(Exception e){
//            e.printStackTrace();
//        }                
//        
    }

    private void saveOut(String output){
//        try {
//            File file = new File(output);
//            FileWriter fw = new FileWriter(file.getAbsoluteFile());
//            BufferedWriter bw = new BufferedWriter(fw);
//            bw.write("; Viterbi-decodings of "+param[1]+" using HMM "+param[0]);
//            bw.newLine();
//            bw.newLine();
//            for(int i=0;i<inSeq.size();i++){
//                bw.write(seqNames.get(i));
//                bw.newLine();
//                //bw.write(inSeq.get(i));
//                //bw.newLine();
//                //bw.write("# ");
////                bw.write(translate(Zlist.get(i)));
//                bw.write(Zlist.get(i));
//                bw.newLine();
////                bw.write("; log P(x,z) (as computed by Viterbi) = "+logprob[i]);
////                bw.newLine();
////                bw.write("; log P(x,z) (as computer by your log-joint-prob) = "+logJP[i]);
////                bw.newLine();
//                bw.newLine();
//            }
//            bw.close();
//        } catch (IOException e) {
//                e.printStackTrace();
//        }               
        try {
            File file = new File(output);
            FileWriter fw = new FileWriter(file.getAbsoluteFile());
            BufferedWriter bw = new BufferedWriter(fw);
            bw.write("; Viterbi-decodings of "+param[1]+" using HMM "+param[0]);
            bw.newLine();
            bw.newLine();
            for(int i=0;i<inSeq.size();i++){
                bw.write(seqNames.get(i));
                bw.newLine();
                //bw.write(inSeq.get(i));
                //bw.newLine();
                //bw.write("# ");
//                bw.write(translate(Zlist.get(i)));
                bw.write(translate(Zlist.get(i)));
                bw.newLine();
//                bw.write("; log P(x,z) (as computed by Viterbi) = "+logprob[i]);
//                bw.newLine();
//                bw.write("; log P(x,z) (as computer by your log-joint-prob) = "+logJP[i]);
//                bw.newLine();
                bw.newLine();
            }
            bw.close();
        } catch (IOException e) {
                e.printStackTrace();
        }               
    }
    
    private String translate(String inp){
        System.out.println("Strarting translation...");
        final char[] map = {'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 
        'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'N'};
        char[] inp_char = inp.toCharArray();
        char[] res = new char[inp_char.length];
        for(int i=0; i<inp_char.length;i++){
            res[i]=map[hidden.get(String.valueOf(inp_char[i]))];
        }
        return(String.valueOf(res));
    }
}
