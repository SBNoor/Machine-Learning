/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package train_by_count19;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;


public class Train_by_count19 {
    char[] obs;
    char[] hid;
    HashMap<String, Integer> hidden = new HashMap<>();
    HashMap<String, Integer> observables = new HashMap<>();
    double[] pi;
    double[][] transitions;
    double[][] emissions;
    ArrayList<String> seqs;
    ArrayList<String> zseqs;
    final static int K = 19;

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        // TODO code application logic here
        new Train_by_count19().Train_by_count19(args);
    }

    public void Train_by_count19(String[] args){
        long startTime = System.currentTimeMillis();
            seqs = new ArrayList<>();
            zseqs = new ArrayList<>();
            readMM("param19state.txt");
            for(int i=0;i<args.length-1;i+=2){
//                if(i!=c){
//                    seqs.add(readInp("../data/genome"+i+".fa"));
//                    zseqs.add(readInp("../data/true-ann"+i+".fa"));
                System.out.println("Reading files: "+args[i]+" and "+args[i+1]);
                    seqs.add(readInp(args[i]));
                    zseqs.add(readInp(args[i+1]));

//                }
            }
            System.out.println("Counting.....");
            count();
            System.out.println("Savind parameters to "+args[args.length-1]);
            save(args[args.length-1]);



        System.out.println("Total time: "+(System.currentTimeMillis()-startTime));
    }
    
    private void readMM(String input){
        try{
            BufferedReader br1 = new BufferedReader(new FileReader(input));
            String line = br1.readLine();
            while(line!=null){
                if(line.contains("hidden")){
                    String[] l = br1.readLine().split(" ");
                    hid = new char[l.length];
                    for(int i=0;i<l.length;i++){
                        hidden.put(l[i], i);
                        hid[i] = l[i].charAt(0);
                    }
                }
                if(line.contains("observables")){
                    String[] l = br1.readLine().split(" ");
                    obs = new char[l.length];
                    for(int i=0;i<l.length;i++){
                        observables.put(l[i], i);
                        obs[i] = l[i].charAt(0);
                    }
                }
                line = br1.readLine();
            }
        }catch(Exception e){
            e.printStackTrace();
        }                
    }

    private String readInp(String input){
        String S=null;
        try{
        S = new String (Files.readAllBytes(Paths.get(input)),Charset.forName("UTF-8"));
            S=S.substring(S.indexOf("\n",S.indexOf(">")));            
            S=S.replace("\r", "");
            S=S.replace("\n", "");
        }catch(Exception e){
            e.printStackTrace();
        }
        return(S);        
    }
    
    private char [] translate_states(char[] seq){
        char[] out = new char[seq.length];
        for(int i=0; i<seq.length; i++){
            if(seq[i]=='N'){
                out[i] = 'x';
            }
            else if(seq[i]=='C'){                
                if(seq[i-1]!='C'){
                    out[i]='a';
                }
                else if(out[i-1]=='g'){
                    out[i]='h';
                }
                else if(out[i-1]=='h'){
                    out[i]='i';
                }               
                else if((i+3)>=seq.length || seq[i+3]!='C'){
                    out[i]='g';
                }
                else if(out[i-1]=='a'){
                    out[i]='b';
                }
                else if(out[i-1]=='b'){
                    out[i]='c';
                }
                else if(out[i-1]=='c' || out[i-1]=='f'){
                    out[i]='d';
                }
                else if(out[i-1]=='d'){
                    out[i]='e';
                }
                else if(out[i-1]=='e'){
                    out[i]='f';
                }
            }
            else if(seq[i]=='R'){
                if(seq[i-1]!='R'){
                    out[i]='j';
                }
                else if(out[i-1]=='p'){
                    out[i]='q';
                }
                else if(out[i-1]=='q'){
                    out[i]='r';
                }                               
                else if(seq[i+3]!='R'){
                    out[i]='p';
                }
                else if(out[i-1]=='j'){
                    out[i]='k';
                }
                else if(out[i-1]=='k'){
                    out[i]='l';
                }
                else if(out[i-1]=='l' || out[i-1]=='o'){
                    out[i]='m';
                }
                else if(out[i-1]=='m'){
                    out[i]='n';
                }
                else if(out[i-1]=='n'){
                    out[i]='o';
                }
            }
        }
        return(out);
    }

    private void count(){
        pi = new double[K];
        transitions = new double[K][K];
        emissions = new double[K][observables.size()];
        for(int m=0;m<seqs.size();m++){
            char[] X = seqs.get(m).toCharArray();
            char[] Z = zseqs.get(m).toCharArray();
            Z = translate_states(Z);

            int N = X.length;
            pi[hidden.get(String.valueOf(Z[0]))]++;
            emissions[hidden.get(String.valueOf(Z[0]))][observables.get(String.valueOf(X[0]))]++;
            for(int n=1;n<N;n++){
                try{
                transitions[hidden.get(String.valueOf(Z[n-1]))][hidden.get(String.valueOf(Z[n]))]++;
                }catch(NullPointerException e){
                    System.out.println(n);
                    System.out.println(Z[n+100]);
                    e.printStackTrace();
                }
                emissions[hidden.get(String.valueOf(Z[n]))][observables.get(String.valueOf(X[n]))]++;
            }
        }
        
        double piSum = getSum(pi);
        for(int k=0;k<K;k++){
            pi[k]=pi[k]/piSum;
        }
        
        for(int j=0;j<K;j++){
            double transSum=getSum(transitions[j]);
            for(int k=0;k<K;k++){
                transitions[j][k]=transitions[j][k]/transSum;
            }
        }
        
        for(int k=0;k<K;k++){
            double emSum = getSum(emissions[k]);
            for(int d=0;d<emissions[k].length;d++){
                emissions[k][d] = emissions[k][d]/emSum;
            }
        }
    }

    private double getSum(double[] in){
        double sum = 0;
        for(double d:in)
            sum+=d;
        return sum;
    }

        private void save(String output){
        try {
            File file = new File(output);
            FileWriter fw = new FileWriter(file.getAbsoluteFile());
            BufferedWriter bw = new BufferedWriter(fw);
            bw.write("#\n");
            bw.write("# 19-state HMM for gene prediction from train_by_count19.\n");
            bw.write("#\n");
            bw.newLine();
            bw.write("hidden\n");
            bw.write(hid[0]);
            for(int i=1;i<hid.length;i++){
                bw.write(" "+hid[i]);
            }
            bw.newLine();
            bw.newLine();
            bw.write("observables\n");
            bw.write(obs[0]);
            for(int i=1;i<obs.length;i++){
                bw.write(" "+obs[i]);
            }
            bw.newLine();
            bw.newLine();
            bw.write("pi\n");
            bw.write(String.valueOf(pi[0]));
            for(int i=1;i<pi.length;i++){
                bw.write(" "+pi[i]);
            }
            bw.newLine();
            bw.newLine();
            bw.write("transitions\n");
            for(int k=0;k<K;k++){
                bw.write(String.valueOf(transitions[k][0]));
                for(int i=1;i<transitions[k].length;i++){
                    bw.write(" "+transitions[k][i]);
                }
                bw.newLine();
            }
            bw.newLine();
            bw.newLine();
            bw.write("emissions\n");
            for(int k=0;k<K;k++){
                bw.write(String.valueOf(emissions[k][0]));
                for(int i=1;i<emissions[k].length;i++){
                    bw.write(" "+emissions[k][i]);
                }
                bw.newLine();
            }
            bw.newLine();
            bw.newLine();
            
            bw.close();
        } catch (IOException e) {
                e.printStackTrace();
        }               
    }


    
}
