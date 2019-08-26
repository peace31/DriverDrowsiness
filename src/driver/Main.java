package driver;
import java.awt.*;  
import java.awt.image.BufferedImage;  
import java.io.ByteArrayInputStream;  
import java.io.IOException;  

import javax.imageio.ImageIO;  
import javax.swing.*;  

import org.opencv.core.Core;  
import org.opencv.core.Mat;  
import org.opencv.core.MatOfByte;  
import org.opencv.core.MatOfRect;  
import org.opencv.core.Point;  
import org.opencv.core.Rect;  
import org.opencv.core.Scalar;  
import org.opencv.core.Size;  
import org.opencv.highgui.Highgui;  
import org.opencv.highgui.VideoCapture;  
import org.opencv.imgproc.Imgproc;  
import org.opencv.objdetect.CascadeClassifier;  
import org.opencv.objdetect.Objdetect;
public class Main {  
    
	public static void main(String arg[]) throws InterruptedException{  
      System.loadLibrary(Core.NATIVE_LIBRARY_NAME); 
      JFrame Fr = new JFrame("Face detection by webcam");  
      Fr.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);  
     
      Fdetection FD=new Fdetection();  
      Fpanel FP = new Fpanel();  
      Fr.setSize(450,450);
      Fr.setBackground(Color.gray);
      Fr.add(FP,BorderLayout.CENTER);       
      Fr.setVisible(true);       
       Mat wimg=new Mat();  
       VideoCapture wc =new VideoCapture(0);   
   
        if( wc.isOpened())  
          {  
           Thread.sleep(400);
           while( true )  
           {  
        	 wc.read(wimg);  
             if( !wimg.empty() )  
              {   
            	  Thread.sleep(200); 
            	  Fr.setSize(wimg.width()+40,wimg.height()+60);  
                  wimg=FD.detect(wimg);  
                  FP.Bimage(wimg);  
                  FP.repaint();   
              }  
              else  
              {   
                   System.out.println("Frame capture loss");   
                   break;   
              }  
             }  
            }
           wc.release();
 
      } 
	
}
class Fpanel extends JPanel{  
     private static final long serialVersionUID = 1L;  
     private BufferedImage Bimg;  
     // Create a constructor method  
     public Fpanel(){  
          super();   
     }  
     /*  
     */       
     public boolean Bimage(Mat matrix) {  
          MatOfByte mb=new MatOfByte();  
          Highgui.imencode(".jpg", matrix, mb);  
          try {  
               this.Bimg = ImageIO.read(new ByteArrayInputStream(mb.toArray()));  
          } catch (IOException e) {  
               e.printStackTrace();  
               return false; // Error  
          }  
       return true; // Successful  
     }  
     public void paintComponent(Graphics g){  
          super.paintComponent(g);   
          if (this.Bimg==null) return;         
           g.drawImage(this.Bimg,10,10,this.Bimg.getWidth(),this.Bimg.getHeight(), null);
     }
        
}  
class Fdetection {  
     private CascadeClassifier FC; 
     CascadeClassifier MC;
     CascadeClassifier EC;
     // Create a constructor method  
     public Fdetection(){  
                 
        FC=new CascadeClassifier("haarcascade_frontalface_alt2.xml"); 
        EC = new CascadeClassifier("haarcascade_eye.xml");
        MC= new CascadeClassifier("haarcascade_mcs_mouth.xml");
     }  
     public Mat detect(Mat INF){  
          Mat rmat=new Mat();  
          Mat gmat=new Mat();  
         
          
          MatOfRect fmat = new MatOfRect();  
          INF.copyTo(rmat);  
          INF.copyTo(gmat);  
          Imgproc.cvtColor( rmat, gmat, Imgproc.COLOR_BGR2GRAY);  
          Imgproc.equalizeHist( gmat, gmat );  
          FC.detectMultiScale(gmat, fmat);  
          System.out.println(String.format("Face Detection: %s	", fmat.toArray().length));  
          int i=0;
          for(Rect size:fmat.toArray())  
          { 
        	  Core.rectangle(rmat,  
        			    new Point(size.x, size.y),  
        			    new Point(size.x + size.width, size.y + size.height), 
        			    new Scalar(255, 0, 0));
        	  
        	  Mat fsubmat = rmat.submat(size);
              MatOfRect emat = new MatOfRect();
              EC.detectMultiScale(fsubmat, emat,5,5,0,new Size(10,10) ,new Size());            
              System.out.println(String.format("Eyes Detection: %s	", emat.toArray().length));
              if(fmat.toArray().length>0 && emat.toArray().length<2)
            	  System.out.println(String.format("Driver is sleeping!"));   
              for (Rect size1:emat.toArray())
              {
            	 
                 Point center1 = new Point(size.x + size1.x + size1.width * 0.5, size.y + size1.y + size1.height * 0.5);
                 int radius = (int) Math.round((size1.width + size1.height) * 0.25);
                 Core.circle(rmat, center1, radius, new Scalar(255, 0, 0), 4, 8, 0);
              }
              i++;
          
          
          MatOfRect mmat = new MatOfRect();
          
          Mat rmat1=new Mat();  
          Mat gmat1=new Mat(); 
        
          INF.copyTo(rmat1);  
          INF.copyTo(gmat1);  
          Imgproc.cvtColor( rmat1, gmat1, Imgproc.COLOR_BGR2GRAY);  
          Imgproc.equalizeHist( gmat1, gmat1 );  
            
       
          MC.detectMultiScale(fsubmat, mmat, 1.1, 2, Objdetect.CASCADE_FIND_BIGGEST_OBJECT  | Objdetect.CASCADE_SCALE_IMAGE, new Size(30, 30), new Size());
          for (Rect mmat1:mmat.toArray()) {
        	  System.out.println(String.format("Mouth Detection: %s	", mmat.toArray().length));
        		  Point center13 = new Point(size.x + mmat1.x + mmat1.width *0.5,
        		                    size.y + mmat1.y + mmat1.height*0.5);
        		 int radius = (int) Math.round(mmat1.width / 2);
                 Core.circle(rmat, center13, radius, new Scalar(255, 0, 0), 4, 8, 0);
          }
          
              }  
          
          return rmat;  
     }  
}  
