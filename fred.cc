#include <iostream>
#include <iomanip>

using namespace std;

#include <RAT/DS/Run.hh>
#include <RAT/DS/PMTInfo.hh>
#include <RAT/DS/Root.hh>
#include <RAT/DS/MC.hh>
#include <RAT/DS/MCParticle.hh>
#include <RAT/DS/EV.hh>
#include <RAT/DS/PMT.hh>
#include <RAT/DS/PathFit.hh>
#include <RAT/DS/BonsaiFit.hh>

#include <TROOT.h>
#include <TFile.h>
#include <TTree.h>
#include <TCanvas.h>
#include <TH1.h>
#include <TH2.h>
#include <TProfile.h>
#include <TApplication.h>

#include <TRandom.h>

#include <pmt_geometry.h>
#include <goodness.h>
#include <searchgrid.h>
#include <fourhitgrid.h>

//Need to seperate the Inner-Detector tubes from the Outer-Detector tubes
static const int innerPMTcode = 1;
static const int vetoPMTcode  = 2;

//static const int triggerThreshold = 9;

#include "ariadne.h"
#include "azimuth_ks.h"
#include "distpmt.h"

//Qfit 
#include <TRandom3.h>

//void ariadne(float *vertex,int *nrhits,float *positions,
//	     float *direction,float *goodness,float *quality,
//	     float *cos_scat,int *nrscat);
extern "C"{
//void lfariadne_(float *av,int *anhit,float *apmt,float *adir, float *amsg, float *aratio,int *anscat,float *acosscat);
void lfariadn2_(float *av,int *anhit,float *apmt,float *adir, float *amsg, float *aratio,int *anscat,float *acosscat);
}

int nwin(RAT::DS::PMTInfo *pmtinfo,
	 float twin,float *v,int nfit, int *cfit, float *tfit, int *cwin);

TVector3 Fitting_Likelihood_Ascent(RAT::DS::Root* ds, RAT::DS::PMTInfo* pmtinfo, RAT::DS::EV *ev ,TH2F *PDF){

  //---- Reconstructed position vector
  TVector3 Position = {-1e9,-1e9,-1e9};
  TRandom3* gRandom = new TRandom3();

  if (ds->GetEVCount()){

    //---- Define some variables. 
    float_t Likelihood_Best = -1e10; float Jump = 1000.0; int iWalk_Max = 70; TVector3 Test_Vertex = {0,0,0}; TVector3 Best_Vertex; float Start = 3000.0; TVector3 End_Vector = {0,0,0}; vector<TVector3> Vector_List = {{0,0,0},{Start,0,0},{-Start,0,0},{0,Start,0},{0,-Start,0},{0,0,Start},{0,0,-Start},{Start,Start,0},{Start,-Start,0},{-Start,Start,0},{-Start,-Start,0},{Start,Start,Start},{Start,-Start,Start},{-Start,Start,Start},{-Start,-Start,Start},{Start,Start,-Start},{Start,-Start,-Start},{-Start,Start,-Start},{-Start,-Start,-Start}};

    //---- Go for a walk...
    for(int iWalk = 0; iWalk < iWalk_Max; iWalk++){
      if (iWalk < 19){Test_Vertex = Vector_List[iWalk];}
      else if (iWalk == 19){gRandom->Sphere(End_Vector[0],End_Vector[1],End_Vector[2],Jump); Test_Vertex = Position + End_Vector;}

      float_t Likelihood = 0;

      for(long iPMT = 0; iPMT < ev->GetPMTCount(); iPMT++ ){
        int PMT_ID             = ev->GetPMT(iPMT)->GetID();
        int pmt_type           = pmtinfo->GetType(PMT_ID);
        if( pmt_type != 1 ) continue;
        TVector3 PMT_Position  = pmtinfo->GetPosition(PMT_ID);
        TVector3 PMT_Direction = pmtinfo->GetDirection(PMT_ID);
        TVector3 R_Test_Vector = Test_Vertex - PMT_Position;
        float_t Angle          = cos(R_Test_Vector.Angle(PMT_Direction));
        Likelihood += ev->GetPMT(iPMT)->GetCharge()*log(PDF->GetBinContent(PDF->FindBin(R_Test_Vector.Mag(),Angle)));
      }

      for (long ipmt = 0; ipmt < pmtinfo->GetPMTCount(); ipmt++){
        int pmt_type           = pmtinfo->GetType(ipmt);
        if( pmt_type != 1 ) continue;
        TVector3 PMT_Position  = pmtinfo->GetPosition(ipmt);
        TVector3 PMT_Direction = pmtinfo->GetDirection(ipmt);
        TVector3 R_Test_Vector = Test_Vertex - PMT_Position;
        float_t Angle          = cos(R_Test_Vector.Angle(PMT_Direction));
        Likelihood -= PDF->GetBinContent(PDF->FindBin(R_Test_Vector.Mag(),Angle));
      }

      //---- If we find a test vertex with a larger likelihood, that is the new reconstructed position 
      if (Likelihood > Likelihood_Best){Likelihood_Best = Likelihood; iWalk--; Jump=Jump/1.05; Position = Test_Vertex;
        if (End_Vector[0] != 0 && End_Vector[1] !=0 && End_Vector[2] !=0){Test_Vertex = Position + End_Vector;}
        else {gRandom->Sphere(End_Vector[0],End_Vector[1],End_Vector[2],Jump); Test_Vertex = Position + End_Vector;}
      }
      else{gRandom->Sphere(End_Vector[0],End_Vector[1],End_Vector[2],Jump); Test_Vertex = Position + End_Vector;}
    }
  }

  //---- After scanning for the number of iterations defined, the position with the largest likelihood gives the reconstructed position
  return Position;
}

int main(int argc, char **argv)
{

  int saveNonTriggeredData = 0;
  float darkNoise,offsetT,minT, maxT,muon_pe_thresh,closestPMT_thresh;
  char do_clusfit,do_QFit,useAngle, do_betas;
  int nsct,detector_threshold;
  int crash_count=0;
  int tot_inner,tot_veto,id, id2;

  printf("\n\nWelcome to FRED (Functions to Reconstruct Events in the Detector). The function can take no less than two input and up\n");
  printf("to 12 inputs: infile,outfile,darkNoise,detector_threshold,time_nX,useAngle,timeOffset,minTime,maxTime,do_clusfit,do_QFIT,do_betas\n\n");
  printf("%d input arguments in function bonsai.\n",argc);
  
  // check if minimum arguments exist
  if (argc<3)
    {
      printf("Less than the required number of arguments\n");
      return -1;
    }
  // set default values
  darkNoise = 3000.; //As agreed for Path A/ Path B comparison
  offsetT   = 800.;
  minT      = -500.;
  maxT      = 1000.;
  useAngle  = 1;
  float time_nX = 9;
  do_clusfit =0;
  do_QFit   = 0;
  do_betas  = 1;

  detector_threshold = 9;

  muon_pe_thresh = 90.; // 15 MeV for 6pe/MeV

  closestPMT_thresh = 700.; // Events must reconstruct 700 mm from PMT surface

  switch(argc)
    {
    case 3:
      printf("Only input file and output file are provided. All other values set to default.\n");
      break;
    case 4:
      printf("Only input file and output file and dark noise rate are provided. All other values set to default.\n");
      darkNoise = float(strtol(argv[3],NULL,10));  
      break;
    case 5:
      // printf("Only input file and output file are provide. All other values set to default.\n");
      darkNoise = float(strtol(argv[3],NULL,10));
      detector_threshold  = atoi(argv[4]);
      break;
    case 6:
      // printf("Only input file and output file are provided. All other values set to default.\n");
      darkNoise = float(strtol(argv[3],NULL,10));
      detector_threshold = atoi(argv[4]);
      time_nX  = float(strtol(argv[5],NULL,10));
      break;
    case 7:
      // printf("Only input file and output file are provide. All other values set to default.\n");
      darkNoise = float(strtol(argv[3],NULL,10));
      detector_threshold = atoi(argv[4]);
      time_nX  = float(strtol(argv[5],NULL,10));
      useAngle =         atoi(argv[6]);//sscanf(argv[6],"%d",&useAngle);
      break;
    case 8:
      // printf("Only input file and output file are provide. All other values set to default.\n");
      darkNoise = float(strtol(argv[3],NULL,10));
      detector_threshold  = atoi(argv[4]);
      time_nX  = float(strtol(argv[5],NULL,10));
      useAngle =         atoi(argv[6]);//sscanf(argv[6],"%d",&useAngle);
      offsetT  = float(strtol(argv[7],NULL,10));
      break;
    case 9:
      darkNoise = float(strtol(argv[3],NULL,10));
      detector_threshold  = atoi(argv[4]);
      time_nX  = float(strtol(argv[5],NULL,10));
      useAngle =         atoi(argv[6]);
      offsetT   = float(strtol(argv[7],NULL,10));
      minT      = float(strtol(argv[8],NULL,10));
      break;
    case 10:
      darkNoise = float(strtol(argv[3],NULL,10));
      detector_threshold  = atoi(argv[4]);
      time_nX  = float(strtol(argv[5],NULL,10));
      useAngle =          atoi(argv[6]);//sscanf(argv[6],"%d",&useAngle);
      offsetT   = float(strtol(argv[7],NULL,10));
      minT      = float(strtol(argv[8],NULL,10));
      maxT      = float(strtol(argv[9],NULL,10));
      break;
    case 11:
      darkNoise = float(strtol(argv[3],NULL,10));
      detector_threshold = atoi(argv[4]);
      time_nX  = float(strtol(argv[5],NULL,10));
      useAngle =          atoi(argv[6]);//sscanf(argv[6],"%d",&useAngle);
      offsetT   = float(strtol(argv[7],NULL,10));
      minT      = float(strtol(argv[8],NULL,10));
      maxT      = float(strtol(argv[9],NULL,10));
      do_clusfit =       atoi(argv[10]);//sscanf(argv[10],"%d",&do_clusfit);
      break;
    case 12:
      darkNoise = float(strtol(argv[3],NULL,10));
      detector_threshold = atoi(argv[4]);
      time_nX   = float(strtol(argv[5],NULL,10));
      useAngle  =         atoi(argv[6]);//sscanf(argv[6],"%d",&useAngle);
      offsetT   = float(strtol(argv[7],NULL,10));
      minT      = float(strtol(argv[8],NULL,10));
      maxT      = float(strtol(argv[9],NULL,10));
      do_clusfit =       atoi(argv[10]);//sscanf(argv[10],"%d",&do_clusfit);
      do_QFit =          atoi(argv[11]);//sscanf(argv[11],"%d",&do_QFit);
      break;
    case 13:
      darkNoise          = float(strtol(argv[3],NULL,10));
      detector_threshold = atoi(argv[4]);
      time_nX            = float(strtol(argv[5],NULL,10));
      useAngle           = atoi(argv[6]);//sscanf(argv[6],"%d",&useAngle);
      offsetT            = float(strtol(argv[7],NULL,10));
      minT               = float(strtol(argv[8],NULL,10));
      maxT               = float(strtol(argv[9],NULL,10));
      do_clusfit         = atoi(argv[10]);//sscanf(argv[10],"%d",&do_clusfit);
      do_QFit            = atoi(argv[11]);//sscanf(argv[11],"%d",&do_QFit);
      muon_pe_thresh     = float(strtol(argv[12],NULL,10));
      break;
    case 14:
      darkNoise          = float(strtol(argv[3],NULL,10));
      detector_threshold = atoi(argv[4]);
      time_nX            = float(strtol(argv[5],NULL,10));
      useAngle           = atoi(argv[6]);//sscanf(argv[6],"%d",&useAngle);
      offsetT            = float(strtol(argv[7],NULL,10));
      minT               = float(strtol(argv[8],NULL,10));
      maxT               = float(strtol(argv[9],NULL,10));
      do_clusfit         = atoi(argv[10]);//sscanf(argv[10],"%d",&do_clusfit);
      do_QFit            = atoi(argv[11]);//sscanf(argv[11],"%d",&do_QFit);
      muon_pe_thresh     = float(strtol(argv[12],NULL,10));
      closestPMT_thresh  = float(strtol(argv[13],NULL,10));
      break;


  }
  
  printf("\n\nUsing:\n");
  printf("(1) Infile \t%20s\n",argv[1]); 
  printf("(2) Outfile \t%20s\n",argv[2]);
  printf("(3) darkNoise \t%20.1f\n", darkNoise); 
  printf("(4) detector_threshold \t%12d\n",detector_threshold);
  printf("(5) time_nX \t%20.1f\n",time_nX);
  printf("(6) useAngle \t%20d\n",useAngle);
  printf("(7) offsetT \t%20.1f\n",offsetT);
  printf("(8) minT \t%20.1f\n",minT); 
  printf("(9) maxT  \t%20.1f\n",maxT);
  printf("(10) do clusft \t %19d\n",do_clusfit);
  printf("(11) do qfit \t %19d\n\n",do_QFit);
  printf("(12) muon_pe_thresh \t %20.1f\n",muon_pe_thresh);
  printf("(12) closestPMT_thresh \t %20.1f\n",closestPMT_thresh);
  printf("(test) do_betas \t%19d\n", do_betas);
 
  TVector3 Best_Fit;
  TFile *fPDF      = new TFile("PDF.root");
  if(!fPDF->IsOpen()){
    if(do_QFit){
      printf("PDF.root file does not exist. Exiting program. Please\ncopy over a PDF.root or turn QFit option off.\n");
      return -1;
    }
  }
  TH2F *PDF    = (TH2F*)fPDF->Get("h_R_Cos_Theta");

  Int_t    gtid=0, mcid=0, code = 0, subid=0, tot_nhit=0, nhits=0, veto_hit=0;
  Int_t    totVHIT=0,inner_hit=0,inner_hit_prev=0,veto_hit_prev=0,inner_hit_next=0,veto_hit_next=0;
  Int_t inner_hit_mid = 0, veto_hit_mid=0;

  Int_t particleCountMC = 0;

  Int_t    nsel=0;
  Double_t n9=0., nX = 0., nOff = 0., n100 = 0., n400 = 0., bonsai_goodness=0., dir_goodness=0., azi_ks;

  Double_t n9_prev = 0., nX_prev = 0., n100_prev = 0., n400_prev = 0., dn9prev = 0., dn100prev = 0.;
  Double_t bonsai_goodness_prev = 0., dir_goodness_prev = 0., azi_ks_prev = 0.;

  Double_t n9_next = 0., nX_next = 0., n100_next = 0., n400_next = 0.;
  Double_t bonsai_goodness_next = 0., dir_goodness_next = 0., azi_ks_next = 0.;

  Double_t n9_mid = 0., nX_mid = 0., n100_mid = 0., n400_mid = 0.;
  Double_t bonsai_goodness_mid = 0., dir_goodness_mid = 0., azi_ks_mid = 0.;

  Double_t clusfit_goodness=0.;

  Double_t totPE=0., innerPE=0., vetoPE=0.;
  Double_t dist_pmt;
  double   wall[3];
  Double_t x=0., y=0., z=0., t=0., u=0., v=0., w=0.;

  Double_t cx=0., cy=0., cz=0., ct=0.;
  Double_t mcx=0., mcy=0., mcz=0., mct=0., mcu=0., mcv=0., mcw=0.;

  Double_t prev_mcx=0., prev_mcy=0., prev_mcz=0., prev_mct=0., prev_mcu=0., prev_mcv=0., prev_mcw=0.;
  Double_t mid_mcx=0., mid_mcy=0., mid_mcz=0., mid_mct=0., mid_mcu=0., mid_mcv=0., mid_mcw=0.;
  Double_t next_mcx=0., next_mcy=0., next_mcz=0., next_mct=0., next_mcu=0., next_mcv=0., next_mcw=0.;

  Double_t closestPMT=0.,closestPMT_prev = 0.,closestPMT_next = 0.,closestPMT_mid = 0.,mc_energy=0.,mc_energy_prev=0.,mc_energy_mid=0.,mc_energy_next=0.;
  Double_t dxpx=0.,dypy=0.,dzpz=0.,drpr=0.,dxnx=0.,dyny=0.,dznz=0.,drnr=0.,dxmcx=0.,dymcy=0.,dzmcz=0.,drmcr=0.;

  Double_t prev_x = -1e9,prev_y= -1e9,prev_z= -1e9, prev_t, prev_u = -1e9,prev_v= -1e9,prev_w= -1e9, p2W,p2ToB;
  Double_t mid_x=0., mid_y=0., mid_z=0., mid_t=0., mid_u=0., mid_v=0., mid_w=0.;
  Double_t next_x = -1e9,next_y= -1e9,next_z= -1e9, next_t, next_u = -1e9,next_v= -1e9,next_w= -1e9;

  //Double_t mid_x = -1e9,mid_y= -1e9,mid_z= -1e9, mid_t;

  Double_t timestamp=0., timestamp0=0., timestamp_prev = 0., timestamp_next = 0., timestamp_mid = 0., dt_sub=0., dt_prev_us=0., dt_next_us=0.;
  Int_t sub_event_tally[20] = {};
  Double_t pmtBoundR=0.,pmtBoundZ=0.;

  Int_t muon_flag = 0;

  // Info for the Legendre Polynomial Angular Fit
  Double_t Dot=0., Cross=0., Magnitude=0., theta=0., costheta=0.;
  Double_t A[3], B[3];
  Double_t beta_one=0., beta_two=0., beta_three=0., beta_four=0.;
  Double_t beta_five=0., beta_six=0., beta_one_prev=0.;
  Double_t beta_two_prev=0., beta_three_prev=0., beta_four_prev=0.;
  Double_t beta_five_prev=0., beta_six_prev=0.;
  vector <double > beta_one_array, beta_two_array,beta_three_array;
  vector < double > beta_four_array, beta_five_array, beta_six_array;

  // dark noise stuff
  TRandom rnd;
  int npmt;
  float darkrate,tmin,tmax;
  int ndark,darkhit;
  int vhit;

  // Ariadne stuff
  float adir[3],agoodn,aqual,cosscat;

  // root stuff
  TFile *f;
  TTree *rat_tree,*run_tree,*data;
  Int_t n_events;
  TTree *run_summary;

  // rat stuff
  RAT::DS::Root *ds=new RAT::DS::Root();
  RAT::DS::Run  *run=new RAT::DS::Run();
  RAT::DS::EV *ev;
  RAT::DS::PMTInfo *pmtinfo;
  RAT::DS::MC *mc;
  RAT::DS::MCParticle *prim;
  RAT::DS::PMT *pmt;
  RAT::DS::PMT *pmt2;

  // BONSAI stuff
  fit_param   bspar;
  bonsaifit   *bsfit,*cffit;
  pmt_geometry *bsgeom;
  likelihood  *bslike;
  goodness    *bsgdn;
  fourhitgrid *bsgrid;
  int         cables[5000],veto_cables[5000],channel[5000];
  int	      cables_win[5000],veto_cables_win[5000];
  float       times[5000],veto_times[5000], hittime[5000];
  float       charges[5000],veto_charges[5000],pmtcharge[5000];
  int         event,sub_event,n,count;
  int         inpmt;
  int         hit,nhit,veto_count;
  float       bonsai_vtxfit[4];
  double      vertex[3],dir[3];
  float       goodn[2];
  float       xyz[6991];
  // likelihood information
  Int_t num_tested;
  Double_t best_like, worst_like, average_like, average_like_05m;

  Double_t xQFit =-999999.99,yQFit=-999999.99, zQFit=-999999.99,closestPMTQFit=-999999.99;
  Double_t xQFit_prev =-999999.99,yQFit_prev=-999999.99, zQFit_prev=-999999.99;
  Double_t xQFit_mid =-999999.99,yQFit_mid=-999999.99, zQFit_mid=-999999.99;
  Double_t xQFit_next =-999999.99,yQFit_next=-999999.99, zQFit_next=-999999.99;



  Double_t closestPMTQFit_prev=-999999.99, closestPMTQFit_next=-999999.99,closestPMTQFit_mid=-999999.99, drrQFit=-999999.99;
  Int_t  QFit = 0;

  rnd.SetSeed();  
  // open input file
  f= new TFile(argv[1]);

  rat_tree=(TTree*) f->Get("T");
  rat_tree->SetBranchAddress("ds", &ds);
  run_tree=(TTree*) f->Get("runT");
  if (rat_tree==0x0 || run_tree==0x0)
    {
      printf("can't find trees T and runT in this file\n");
      return -1;
    }
  run_tree->SetBranchAddress("run", &run);
  if (run_tree->GetEntries() != 1)
    {
      printf("More than one run! Ignoring all, but the geometry for the first run\n");
      //return -1;
    }

  // open output file
  TFile *out=new TFile(argv[2],"RECREATE");

  data=new TTree("data","low-energy detector triggered events");

  int time_nX_int = static_cast<int>(time_nX);

  //Define the Integer Tree Leaves
  data->Branch("gtid",&gtid,"gtid/I");
  data->Branch("mcid",&mcid,"mcid/I");
  data->Branch("code",&code,"code/I"); // ibd/fast-neutron/accidental/radio-nuclide
  data->Branch("muon_flag",&muon_flag,"muon_flag/I");
  data->Branch("subid",&subid,"subid/I");
  //data->Branch("nhit",&nhits,"nhit/I");
  data->Branch("inner_hit_prev",&inner_hit_prev,"inner_hit_prev/I");//inner detector
  data->Branch("inner_hit",&inner_hit_mid,"inner_hit/I");//inner detector    
  data->Branch("inner_hit_next",&inner_hit_next,"inner_hit_next/I");//inner detector
  if (do_clusfit)
    data->Branch("ncherenkovhit",&nsel,"ncherenkovhit/I");// # of selected hits
  data->Branch("id_plus_dr_hit",&tot_nhit,"id_plus_dr_hit/I");//Inner detector plus dark rate hits
  data->Branch("pmtcharge",&pmtcharge,"pmtcharge[id_plus_dr_hit]/F");
  data->Branch("channel",&channel,"channel[id_plus_dr_hit]/I");
  data->Branch("hittime",&hittime,"hittime[id_plus_dr_hit]/F");
  data->Branch("veto_hit",&veto_hit,"veto_hit/I");//veto detector
  data->Branch("veto_plus_dr_hit",&totVHIT,"veto_plus_dr_hit/I");//veto detector plus dark rate hits  
  data->Branch("veto_hit_prev",&veto_hit_prev,"veto_hit_prev/I");//veto detector
  //Define the double Tree Leaves
  data->Branch("pe",&totPE,"pe/D");
  data->Branch("innerPE",&innerPE,"innerPE/D");
  data->Branch("vetoPE",&vetoPE,"vetoPE/D");
  if(do_QFit){
   data->Branch("xQFit",&xQFit,"xQFit/D"); data->Branch("yQFit",&yQFit,"yQFit/D");
   data->Branch("zQFit",&zQFit,"zQFit/D");data->Branch("QFit",&QFit,"QFit/I");
   data->Branch("closestPMTQFit",&closestPMTQFit,"closestPMTQFit/D");
   data->Branch("closestPMTQFit_prev",&closestPMTQFit_prev,"closestPMTQFit_prev/D");
  }		
  data->Branch("n9_prev",&n9_prev,"n9_prev/D");
  data->Branch("n9",      &n9_mid,"n9/D");
  data->Branch("n9_next",&n9_next,"n9_next/D");
  data->Branch("nOff",    &nOff,"nOff/D");
  data->Branch("n100_prev",&n100_prev,"n100_prev/D");
  data->Branch("n100",      &n100_mid,"n100/D");
  data->Branch("n100_next",&n100_next,"n100_next/D");
  data->Branch("n400_prev",&n400_prev,"n400_prev/D");
  data->Branch("n400",&n400_mid,"n400/D");
  data->Branch("n400_next",&n400_next,"n400_next/D");
  data->Branch("nX_prev",&nX_prev,"nX_prev/D");
  data->Branch("nX",&nX_mid,"nX/D");
  data->Branch("nX_next",&nX_next,"nX_next/D");
  data->Branch("good_pos_prev",&bonsai_goodness_prev,"good_pos_prev/D");
  data->Branch("good_pos",&bonsai_goodness_mid,"good_pos/D");
  data->Branch("good_pos_next",&bonsai_goodness_next,"good_pos_next/D");

//  data->Branch("cables_win",&cables_win,"cables_win[n9]/I");

  if (do_clusfit)
    data->Branch("good_cpos",&clusfit_goodness,"good_cpos/D");

  data->Branch("good_dir_prev",&dir_goodness_prev,"good_dir_prev/D");
  data->Branch("good_dir",&dir_goodness_mid,"good_dir/D");
  data->Branch("good_dir_next",&dir_goodness_next,"good_dir_next/D");

  data->Branch("x",&mid_x,"x/D"); data->Branch("y",&mid_y,"y/D");
  data->Branch("z",&mid_z,"z/D"); data->Branch("t",&mid_t,"t/D");
  if (do_clusfit)
    {
      data->Branch("cx",&cx,"cx/D"); data->Branch("cy",&cy,"cy/D");
      data->Branch("cz",&cz,"cz/D"); data->Branch("ct",&ct,"ct/D");
    }
  data->Branch("u",&mid_u,"u/D"); data->Branch("v",&mid_v,"v/D");
  data->Branch("w",&mid_w,"w/D");

  data->Branch("azimuth_ks_prev",&azi_ks_prev,"azimuth_ks_prev/D");
  data->Branch("azimuth_ks",&azi_ks_mid,"azimuth_ks/D");
  data->Branch("azimuth_ks_next",&azi_ks_next,"azimuth_ks_next/D");

  data->Branch("distpmt",&dist_pmt,"distpmt/D");
  data->Branch("mc_energy",&mc_energy_mid,"mc_energy/D");
  data->Branch("mcx",&mid_mcx,"mcx/D"); data->Branch("mcy",&mid_mcy,"mcy/D");
  data->Branch("mcz",&mid_mcz,"mcz/D"); data->Branch("mct",&mid_mct,"mct/D");
  data->Branch("mcu",&mid_mcu,"mcu/D"); data->Branch("mcv",&mid_mcv,"mcv/D");
  data->Branch("mcw",&mid_mcw,"mcw/D");
  // data->Branch("code",&code,"code/I");
  data->Branch("closestPMT_prev",&closestPMT_prev,"closestPMT_prev/D");//Proximity to PMT wall
  data->Branch("closestPMT",&closestPMT_mid,"closestPMT/D");//Proximity to PMT wall
  data->Branch("closestPMT_next",&closestPMT_next,"closestPMT_next/D");//Proximity to PMT wall

  data->Branch("dxPrevx",&dxpx,"dxPrevx/D");
  data->Branch("dyPrevy",&dypy,"dyPrevy/D");
  data->Branch("dzPrevz",&dzpz,"dzPrevz/D");
  data->Branch("drPrevr",&drpr,"drPrevr/D");

  data->Branch("dxNextx",&dxnx,"dxNextx/D");
  data->Branch("dyNexty",&dyny,"dyNexty/D");
  data->Branch("dzNextz",&dznz,"dzNextz/D");
  data->Branch("drNextr",&drnr,"drNextr/D");
  if(do_QFit){
    data->Branch("drPrevrQFit",&drrQFit,"drPrevrQFit/D");
  }
  data->Branch("dxmcx",&dxmcx,"dxmcx/D");
  data->Branch("dymcy",&dymcy,"dymcy/D");
  data->Branch("dzmcz",&dzmcz,"dzmcz/D");
  data->Branch("drmcr",&drmcr,"drmcr/D");

  data->Branch("dt_sub", &dt_sub, "dt_sub/D"); //time of the sub-event trigger from start of the event mc
  data->Branch("dt_prev_us",&dt_prev_us,"dt_prev_us/D"); //global time between consecutive events in us
  data->Branch("dt_next_us",&dt_next_us,"dt_next_us/D"); //global time between consecutive events in us

  data->Branch("timestamp",&timestamp,"timestamp/D"); //trigger time of sub event from start of run

  // likelihood information from bonsai
  data->Branch("num_tested",&num_tested,"num_tested/I"); // number of tested points
  data->Branch("best_like",&best_like,"best_like/D"); // the best log likelihood
  data->Branch("worst_like",&worst_like,"worst_like/D"); // the worst log likelihood
  data->Branch("average_like",&average_like,"average_like/D"); // the total average log likelihood
  data->Branch("average_like_05m",&average_like_05m,"average_like_05m/D"); // the average log likelihood excluding a 0.5m sphere around the best fit

  // Legendre Isotropy Variables
  if (do_betas){
    data->Branch("beta_one",&beta_one,"beta_one/D");
    data->Branch("beta_two",&beta_two,"beta_two/D");
    data->Branch("beta_three",&beta_three,"beta_three/D");
    data->Branch("beta_four",&beta_four,"beta_four/D");
    data->Branch("beta_five",&beta_five,"beta_five/D");
    data->Branch("beta_six",&beta_six,"beta_six/D");
    data->Branch("beta_one_prev",&beta_one_prev,"beta_one_prev/D");
    data->Branch("beta_two_prev",&beta_two_prev,"beta_two_prev/D");
    data->Branch("beta_three_prev",&beta_three_prev,"beta_three_prev/D");
    data->Branch("beta_four_prev",&beta_four_prev,"beta_four_prev/D");
    data->Branch("beta_five_prev",&beta_five_prev,"beta_five_prev/D");
    data->Branch("beta_six_prev",&beta_six_prev,"beta_six_prev/D");
  }

  run_summary=new TTree("runSummary","mc run summary");
  run_summary->Branch("nEvents",&n_events,"nEvents/I");
  run_summary->Branch("subEventTally",sub_event_tally,"subEventTally[20]/I");
  run_summary->Branch("X",&time_nX_int,"X/I"); //Produce plot with labels showing time window
  run_summary->Branch("muon_pe_thresh",&muon_pe_thresh,"muon_pe_thresh/F");
  run_summary->Branch("closestPMT_thresh",&closestPMT_thresh,"closestPMT_thresh/F");
  run_summary->Branch("darkNoise",&darkNoise,"darkNoise/F"); 
  run_summary->Branch("detector_threshold",&detector_threshold,"detector_threshold/I");
  run_summary->Branch("time_nX",&time_nX,"time_nX/F");
  run_summary->Branch("useAngle",&useAngle,"useAngle/B");
  run_summary->Branch("offsetT",&offsetT,"offsetT/F");
  run_summary->Branch("minT",&minT,"minT/F"); 
  run_summary->Branch("maxT",&maxT,"maxT/F");
  run_summary->Branch("do_clusfit",&do_clusfit,"do_clusfit/B");
  run_summary->Branch("do_QFit",&do_QFit,"do_QFit/B");
  run_summary->Branch("do_betas",&do_betas,"do_betas/B");
  run_summary->Branch("xyz",&xyz,"xyz[6991]/F");
  run_tree->GetEntry(0);


// loop over PMTs and find positions and location of PMT support
  pmtinfo=run->GetPMTInfo();
  n=pmtinfo->GetPMTCount();
  tot_inner = 0; tot_veto =0;
 
  //Determines the number of inner and veto pmts
  for(hit=count=0; hit<n; hit++)
    {
      if (pmtinfo->GetType(hit)==innerPMTcode)     ++tot_inner;
      else if (pmtinfo->GetType(hit)==vetoPMTcode) ++tot_veto;
      else
	printf("PMT does not have valid identifier: %d \n",
	       pmtinfo->GetType(hit));
    }
  if (n != (tot_inner+tot_veto))
    printf("Mis-match in total PMT numbers: %d, %d \n",n, tot_inner+tot_veto);
    
  inpmt= tot_inner;

  // generate BONSAI geometry object
  {
//    float xyz[3*inpmt+1];

    printf("In total there are  %d PMTs in WATCHMAN\n",n);
    
    for(hit=0; hit<n; hit++)
      {
	if(pmtinfo->GetType(hit)==innerPMTcode)
	  {
	    TVector3 pos=pmtinfo->GetPosition(hit);
	    xyz[3*count]=pos[0]*0.1;
	    xyz[3*count+1]=pos[1]*0.1;
	    xyz[3*count+2]=pos[2]*0.1;
	    if (pos[0]>pmtBoundR) pmtBoundR = pos[0];
	    if (pos[2]>pmtBoundZ) pmtBoundZ = pos[2];
	    ++count;
	  }
      }
    
    printf("There are %d inner pmts and %d veto pmts \n ",tot_inner,tot_veto);
    printf("Inner PMT boundary (r,z):(%4.1f mm, %4.1f mm)\n",pmtBoundR,pmtBoundZ);

    if (count!= tot_inner)
      printf("There is a descreptancy in inner PMTS %d vs %d",count,tot_inner);

    // create BONSAI objects from the PMT position array
    bsgeom=new pmt_geometry(inpmt,xyz);
    bslike=new likelihood(bsgeom->cylinder_radius(),bsgeom->cylinder_height());
    bsfit=new bonsaifit(bslike);
  }

  n_events = rat_tree->GetEntries();
  //run_summary->Fill();
  
  // loop over all events
  for (event = 0; event < n_events; event++)
    {
      if (event%10000==0)
	printf("Evaluating event %d of %d (%d sub events)\n",event,n_events,
	       ds->GetEVCount());
      rat_tree->GetEntry(event);


       //Int_t particleCountMC = mc->GetMCParticleCount();
       /*for (int mcP =0; mcP < particleCountMC; mcP++) {
            RAT::DS::MCParticle *prim = mc->GetMCParticle(mcP);
            printf("%4d momentum  : %8.3f %8.3f %8.3f\n", prim->GetPDGCode(), prim->GetMomentum().X(), prim->GetMomentum().Y(), prim->GetMomentum().Z());

        }*/
        //Get the direction of the neutrino. Saved as last particle
        //RAT::DS::MCParticle *prim = mc->GetMCParticle(particleCountMC-1);
        //mcmomv_nu=prim->GetMomentum();
        //dirNu =  prim->GetMomentum();
        //mc_nu_energy = prim->ke;
        //hNuE->Fill(mc_nu_energy);

        //interaction_type = 0.0;
        //Int_t ES_true = 0 , IBD_true = 0 , CC_true =  0 ,ICC_true = 0 , NC_true = 0;


        /* if(particleCountMC ==2 && mc->GetMCParticle(0)->GetPDGCode()==-11 && mc->GetMCParticle(1)->GetPDGCode()==2112){
            printf("IBD Interaction      ... ");
            //ibd+=1;
            //IBD_true =1;
            //interaction_type = 2;
            //RAT::DS::MCParticle *prim = mc->GetMCParticle(0);
            //mc_energy = prim->ke;

            //hNuP->Fill(prim->ke);
            //mcmomv_particle = prim->GetMomentum();
            //totMom = sqrt(pow(prim->GetMomentum().X(),2) +pow(prim->GetMomentum().Y(),2) + pow(prim->GetMomentum().Z(),2));
            //dirTruth =  TVector3(prim->GetMomentum().X()/totMom,prim->GetMomentum().Y()/totMom,prim->GetMomentum().Z()/totMom);
            //posTruth = prim->GetPosition();

        }*/




      sub_event_tally[ds->GetEVCount()]++;
      // loop over all subevents
      for(sub_event=0;sub_event<ds->GetEVCount();sub_event++)
	{

            // reset output variables
          nOff = -999999;
          n9 =  -999999;
          n100 = -999999;
          n400 = -999999;
          nX =  -999999;
          nsel = -999999;
          bonsai_goodness = dir_goodness = x = y = z = t  = u = v = w = azi_ks = dist_pmt = closestPMT = -999999.99;
          clusfit_goodness=cx=cy=cz=ct=-999999.99;
          dxnx = dyny = dznz = drnr = dxpx = dypy = dzpz = drpr =dxmcx = dymcy = dzmcz = drmcr = drrQFit = closestPMTQFit = -999999.99;
          beta_one = beta_two = beta_three = beta_four = beta_five = beta_six = 0.;

	  gtid += 1;
	  mcid = event;
	  subid = sub_event;
	  ev = ds->GetEV(sub_event);
	  totPE = ev->GetTotalCharge();

	  TVector3 temp;
      
	  mc = ds->GetMC();
	  

          particleCountMC = mc->GetMCParticleCount();
          //printf("Particle count: %d\n",particleCountMC);

          for (int mcP =0; mcP < particleCountMC; mcP++) {
            RAT::DS::MCParticle *prim = mc->GetMCParticle(mcP);
            //printf("PID/energy/time/endTime  %4d %4f %4f %4f\n", prim->GetPDGCode(),prim->GetKE(),prim->GetTime()),prim->GetEndTime();
          }  


          if(particleCountMC ==2 && mc->GetMCParticle(0)->GetPDGCode()==-11 && mc->GetMCParticle(1)->GetPDGCode()==2112){
             code = 1;         
          }else if(particleCountMC ==2 && mc->GetMCParticle(1)->GetPDGCode()==-11 && mc->GetMCParticle(0)->GetPDGCode()==2112){
             code = 1;
          }else if(mc->GetMCParticle(0)->GetPDGCode()==2112 && mc->GetMCParticle(0)->GetKE()>10){ 
             code = 2;
          }else{
             code = 4;
          }

          prim = mc->GetMCParticle(0);

         
  	  mc_energy = prim->GetKE();
          if (!subid){
	    temp = prim->GetPosition();
	    mcx = temp.X();
	    mcy = temp.Y();
	    mcz = temp.Z();
            mct = prim->GetTime(); //true emission time
          }
          else{
            temp = prim->GetEndPosition();
            mcx = temp.X();
            mcy = temp.Y();
            mcz = temp.Z();
            mct = prim->GetEndTime();
          }
          // get true event timings
          // times are in ns unless specified
          timestamp = 1e6*mc->GetUTC().GetSec() + 1e-3*mc->GetUTC().GetNanoSec() + 1e-3*ev->GetCalibratedTriggerTime() - 1e6*run->GetStartTime().GetSec()-1e-3*run->GetStartTime().GetNanoSec(); //global time of subevent trigger (us)
          dt_prev_us = timestamp-prev_t; //time since the previous trigger (us)
          dt_sub = ev->GetCalibratedTriggerTime(); //trigger time (first pmt hit time) from start of event mc

	  nhit=ev->GetPMTCount();

          // loop over all PMT hits for each subevent
	  innerPE=0;vetoPE=0;    
	  for(hit=count=veto_count=0; hit<nhit; hit++)
	    {
	      pmt=ev->GetPMT(hit);
	      id = pmt->GetID();
	      //only use information from the inner pmts
	      if(pmtinfo->GetType(id) == innerPMTcode)
		{
		  cables[count]=pmt->GetID()+1;
		  times[count]=pmt->GetTime()+offsetT;
		  charges[count]=pmt->GetCharge();
		  innerPE += pmt->GetCharge();
		  pmtcharge[count]=pmt->GetCharge();
		  channel[count]=pmt->GetID();
		  hittime[count]=pmt->GetTime()+offsetT;
		  count++;
		}
	      else if(pmtinfo->GetType(id)== vetoPMTcode)
		{
		  veto_cables[veto_count]=pmt->GetID()+1;
		  veto_times[veto_count]=pmt->GetTime()+offsetT;
		  veto_charges[veto_count]=pmt->GetCharge();
		  vetoPE += pmt->GetCharge();    
		  veto_count++;
		}
	      else
		printf("Unidentified PMT type: (%d,%d) \n",count,pmtinfo->GetType(id));
	    } // end of loop over all PMT hits
	  veto_hit = veto_count;
	  inner_hit = count;
	  nhit = count;
          //printf("nhit: %d\n",nhit);
          if (inner_hit<detector_threshold){
            //  printf("Event did not pass trigger threshold (%d:%d)\n",inner_hit,triggerThreshold);
              if(saveNonTriggeredData == 1){
                  data->Fill();
              }
              continue;
          }

	  //Inner PMT Dark Rate
	  npmt=tot_inner;
	  darkrate=darkNoise*npmt;
	  tmin=minT+offsetT;
	  tmax= maxT+offsetT;
	  ndark=rnd.Poisson((tmax-tmin)*1e-9*darkrate);

	  //int inhit= count;
	  //loop over (randomly generated) dark hits and
	  //assign random dark rate where event rate is 
	  //below dark rate for the inner detector
	  for(darkhit=0; darkhit<ndark; darkhit++)
	    {
	      int darkcable= (int)(npmt*rnd.Rndm()+1);
	      float darkt=tmin+(tmax-tmin)*rnd.Rndm();
	      // loop over all inner PMT hits
	      for(hit=0; hit<nhit; hit++)
		if (cables[hit]==darkcable) break;
	      if (hit==nhit)
		{
		  cables[hit]=darkcable;
		  times[hit]=darkt;
		  charges[hit]=1;
		  pmtcharge[hit]=rnd.Gaus(1.06, .312);
		  while (pmtcharge[hit]<0){
		  pmtcharge[hit]=rnd.Gaus(1.06, .312);
		  }
		  channel[hit]=darkcable-1;
		  hittime[hit]=darkt;
		  nhit++;
		} 
	      else
		{
		  if (darkt<times[hit]) times[hit]=darkt;
		  charges[hit]++;
		} //end of loop over all inner  PMT hits
	    } // end of loop over inner dark hits
	  //Veto PMT
	  //Inner PMT Dark Rate
      
	  npmt=tot_veto;
	  darkrate=darkNoise*npmt;
	  ndark=rnd.Poisson((tmax-tmin)*1e-9*darkrate);
	  vhit= veto_count;
      //loop over (randomly generated) dark hits and
      //assign random dark rate where event rate is
      //below dark rate for the veto detector
	  for(darkhit=0; darkhit<ndark; darkhit++)
	    {
	      int darkcable=(int) (npmt*rnd.Rndm()+1);
	      float darkt=tmin+(tmax-tmin)*rnd.Rndm();
        // loop over all inner PMT hits
	      for(hit=0; hit<vhit; hit++)
		if (veto_cables[hit]==darkcable) break;
	      if (hit==vhit)
		{
		  veto_cables[hit]=darkcable;
		  veto_times[hit]=darkt;
		  veto_charges[hit]=1;
		  vhit++;
		}
	      else
		{
		  if (darkt<veto_times[hit]) veto_times[hit]=darkt;
		  veto_charges[hit]++;
		} //end of loop over all veto  PMT hits
	    } // end of loop over vetp dark hits 
	  totVHIT= vhit;
	  tot_nhit= nhit;
	  //Determines how many events before crash
	  crash_count++;

	  // generate BONSAI objects
	  bsgdn=new goodness(bslike->sets(),bslike->chargebins(),
			     bsgeom,nhit,cables,times,charges);
	  nsel=bsgdn->nselected();
	  if(nsel<4) {
	    // four selected hits required to form a fourhitgrid.
	    // we will crash if we continue.
	              //Perform Qfit
            if(do_QFit ==1){
              Best_Fit = Fitting_Likelihood_Ascent(ds, pmtinfo, ev, PDF);
              //printf("Qfit: %f",Best_Fit[0]);
              xQFit = Best_Fit[0];
              yQFit = Best_Fit[1];
              zQFit = Best_Fit[2];
              QFit  = 1;

              // calculate smallest distance to any pmt
              p2W = pmtBoundR-sqrt(xQFit*xQFit+yQFit*yQFit);
              p2ToB = pmtBoundZ-sqrt(zQFit*zQFit);
              closestPMTQFit = TMath::Min(p2W,p2ToB);

              // calculate distance from previous subevent
              drrQFit = sqrt(pow(xQFit-xQFit_prev,2)+pow(yQFit-yQFit_prev,2)+pow(zQFit-zQFit_prev,2));
            }
            if(saveNonTriggeredData == 1){
                data->Fill();
            }
            QFit  = 0 ;
            xQFit = yQFit = zQFit = -999999.0;
	    continue;
	  }
	  bsgrid=new fourhitgrid(bsgeom->cylinder_radius(),
				 bsgeom->cylinder_height(),bsgdn);
	  bslike->set_hits(bsgdn);
	  if (do_clusfit)
	    {
	      // Clusfit
	      cffit=new bonsaifit(bsgdn);
	      bsgdn->maximize(cffit,bsgrid);
	      cx=10.*cffit->xfit();
	      cy=10.*cffit->yfit();
	      cz=10.*cffit->zfit();
	      ct=10.*bsgdn->get_zero()-offsetT;
	      clusfit_goodness=cffit->maxq();
	      delete cffit;
	    }
	  // fit
          bool use_cherenkov_angle = true;
          if(useAngle == 0) use_cherenkov_angle = false;
	  bslike->maximize(bsfit, bsgrid, use_cherenkov_angle);

	  x= 10.*bsfit->xfit();
          y= 10.*bsfit->yfit();
          z= 10.*bsfit->zfit();

          if (do_betas){
            if (beta_one_array.size()>99){
              // limit the vector size
              beta_one_array.erase(beta_one_array.begin());
              beta_two_array.erase(beta_two_array.begin());
	      beta_three_array.erase(beta_three_array.begin());
	      beta_four_array.erase(beta_four_array.begin());
	      beta_five_array.erase(beta_five_array.begin());
  	      beta_six_array.erase(beta_six_array.begin());
            }
            //calculate the legendre polynomial coefficients
            for (int i=0;i<inner_hit-1;i++)
              {
              pmt=ev->GetPMT(i);
              id = pmt->GetID();
              TVector3 posA = pmtinfo->GetPosition(id);
              A[0]=posA[0], A[1]=posA[1], A[2] = posA[2];
              A[0]-=x; A[1]-=y; A[2]-=z;

              for (int j=i+1;j<inner_hit;j++)
                {
                pmt2 = ev->GetPMT(j);
                id2 =  pmt2->GetID();
                TVector3 posB = pmtinfo->GetPosition(id2);
                B[0] = posB[0], B[1] = posB[1], B[2] = posB[2];
                B[0]-=x;B[1]-=y;B[2]-=z;
                Dot         = A[0]*B[0]+A[1]*B[1]+A[2]*B[2];
                Cross       = (A[1]*B[2]-A[2]*B[1])+(A[0]*B[2]-A[2]*B[0])+(A[0]*B[1]-A[1]*B[0]);
                Magnitude   = sqrt(Cross*Cross);
                theta       = atan2(Magnitude,Dot); // angle between PMT hits in radians
                costheta = cos(theta);
                beta_one          += costheta;
                beta_two          += 3.*pow(costheta,2) - 1. ;
                beta_three        += 5.*pow(costheta,3) - 3.*costheta ;
                beta_four         += 35.*pow(costheta,4) - 30.*pow(costheta,2) + 3;
                beta_five         += 63.*pow(costheta,5) - 70.*pow(costheta,3) + 15.*costheta;
                beta_six          += 231.*pow(costheta,6) - 315.*pow(costheta,4) + 105.*pow(costheta,2) - 5.;
              }
            }

            //calculate the isotropy parameter for each particle

            beta_one *= 2./float(nhit*(nhit-1));
            beta_two *= 2./float(2.*nhit*(nhit-1));
            beta_three *= 2./float(2.*nhit*(nhit-1));
            beta_four *= 2./float(8.*nhit*(nhit-1));
            beta_five *= 2./float(8.*nhit*(nhit-1));
            beta_six *= 2./float(16.*nhit*(nhit-1));

	    // create array of beta coefficients
            beta_one_array.push_back( beta_one);
            beta_two_array.push_back( beta_two);
            beta_three_array.push_back( beta_three);
            beta_four_array.push_back( beta_four);
            beta_five_array.push_back( beta_five);
            beta_six_array.push_back( beta_six);
          }


	  // calculate n9 and goodness
	  *bonsai_vtxfit=bsfit->xfit();
	  bonsai_vtxfit[1]=bsfit->yfit();
	  bonsai_vtxfit[2]=bsfit->zfit();
          n9 =  bslike->nwind(bonsai_vtxfit,-3,6);
          nOff = bslike->nwind(bonsai_vtxfit,-150,-50);
          n100 =  bslike->nwind(bonsai_vtxfit,-10,90);
          n400 =  bslike->nwind(bonsai_vtxfit,-10,390);
          if (time_nX == 9.) {
            nX = bslike->nwind(bonsai_vtxfit,-3,6);
          }
          else {
            nX =  bslike->nwind(bonsai_vtxfit,-10,time_nX-10); //Stephen Wilson - how to decide the time interval
          }
          bslike->ntgood(bonsai_vtxfit,0,goodn[0]);


          // In the new version where the next variable is read in, the mc event info must be for the previous event
          
          //pt = t
          //pmc_u = mc_u;


          // get the reconstructed emission time
          t=bslike->get_zero()-offsetT;

          float ave;
          int nfit;

          nfit=bslike->nfit();
          num_tested = nfit;
          best_like = bslike->get_ll();
          worst_like = bslike->worstquality();

          nfit=bslike->average_quality(ave,bonsai_vtxfit,-1);
          average_like= ave;
          nfit=bslike->average_quality(ave,bonsai_vtxfit,50);
          average_like_05m = ave;

	  // get momentum vector and normalize it
	  temp = prim->GetMomentum();
	  temp = temp.Unit();
	  mcu = temp.X();mcv = temp.Y();mcw = temp.Z();
	  tot_nhit = nhit;
          
          // Fill in value from last event

          dxpx = dxnx; 
          dypy = dyny;
          dzpz = dznz;
          drpr = drnr;
          
	  // get distance and time difference to previous fit event
	  dxnx = x-prev_x;dyny = y-prev_y;dznz = z-prev_z;
	  drnr = sqrt(dxnx*dxnx+dyny*dyny+dznz*dznz);
          if(dxnx>1e6)
	    {
	      dxnx = 0.;dyny = 0.;dznz = 0.; drnr = 0.; // Why is this here??
	    }
	  //dxmcx = x-mcx;dymcy = y-mcy;dzmcz = z-mcz;
          //drmcr = sqrt(dxmcx*dxmcx+dymcy*dymcy+dzmcz*dzmcz);

	  // calculate smallest distance to any pmt
	  p2W = pmtBoundR-sqrt(x*x+y*y);
	  p2ToB = pmtBoundZ-sqrt(z*z);
	  closestPMT = TMath::Min(p2W,p2ToB);

	  // do direction fit
	  bonsai_goodness = goodn[0];
	  // find all PMTs within 9 nsec window
	  int n9win=nwin(pmtinfo,9,bonsai_vtxfit,nhit,cables,times,cables_win);
	  float apmt[3*n9win];
	  // fill PMT positions into an array
	  for(hit=0; hit<n9win; hit++)
	    {

	      TVector3 n9pos=pmtinfo->GetPosition(cables_win[hit]-1);
	      apmt[3*hit]=n9pos.X()*0.1;
	      apmt[3*hit+1]=n9pos.Y()*0.1;
	      apmt[3*hit+2]=n9pos.Z()*0.1;
	    }
	  // call direction fit and save results
	  adir[0]=adir[1]=adir[2]=-2;
	  {
	    ariadne ari(bonsai_vtxfit,n9win,apmt,0.719);
	    ari.fit();
	    agoodn=ari.dir_goodness();
	    if (agoodn>=0) ari.dir(adir);
	    dir_goodness = agoodn;
	    u=adir[0];
	    v=adir[1];
	    w=adir[2];
	  }
	  azi_ks=azimuth_ks(n9win,apmt,bonsai_vtxfit,adir);
	  vertex[0]=x;
	  vertex[1]=y;
	  vertex[2]=z;
	  dir[0]=adir[0];
	  dir[1]=adir[1];
	  dir[2]=adir[2];
	  if ((dir[0]*dir[0]+dir[1]*dir[1]+dir[2]*dir[2]>1.00001) ||
	      (dir[0]*dir[0]+dir[1]*dir[1]+dir[2]*dir[2]<0.99999))
	    dist_pmt=-1;
	  else
	    dist_pmt=distpmt(vertex,dir,pmtBoundR,pmtBoundZ,wall);
	   
          //Perform Qfit
          if(do_QFit ==1){
            Best_Fit = Fitting_Likelihood_Ascent(ds, pmtinfo, ev, PDF);
            //printf("Qfit: %f",Best_Fit[0]);
            xQFit = Best_Fit[0];
            yQFit = Best_Fit[1];
            zQFit = Best_Fit[2];
            QFit  = 1;
          
            // calculate smallest distance to any pmt
            p2W = pmtBoundR-sqrt(xQFit*xQFit+yQFit*yQFit);
            p2ToB = pmtBoundZ-sqrt(zQFit*zQFit);
            closestPMTQFit = TMath::Min(p2W,p2ToB);

            // calculate distance from previous subevent
            drrQFit = sqrt(pow((xQFit-xQFit_prev),2)+pow((yQFit-yQFit_prev),2)+pow((zQFit-zQFit_prev),2));
          }

          if( closestPMT > closestPMT_thresh) {

            dxmcx = x-mcx;dymcy = y-mcy;dzmcz = z-mcz;
            drmcr = sqrt(dxmcx*dxmcx+dymcy*dymcy+dzmcz*dzmcz);

            dxpx = mid_x  - prev_x; dypy =  mid_y - prev_y; dzpz =  mid_z - prev_z;
            dxnx = next_x -  mid_x; dyny = next_y -  mid_y; dznz = prev_z -  mid_z ;

            drnr = sqrt(dxnx*dxnx+dyny*dyny+dznz*dznz);
            drpr = sqrt(dxpx*dxpx+dypy*dypy+dzpz*dzpz);

            dt_prev_us = timestamp_mid - timestamp_prev;
            dt_next_us = timestamp_next - timestamp_mid;


	    data->Fill();
            //save reference values for the next subevent
          
            prev_x = mid_x;
            prev_y = mid_y;
            prev_z = mid_z;
            prev_u = mid_u;
            prev_v = mid_v;
            prev_w = mid_w;
            timestamp_prev = timestamp_mid;
            //
            prev_mcx = mid_mcx;
            prev_mcy = mid_mcy;
            prev_mcz = mid_mcz;
            prev_mcu = mid_mcu;
            prev_mcv = mid_mcv;
            prev_mcw = mid_mcw;
            prev_mct = mid_mct;
            mc_energy_prev = mc_energy_mid;
            //
            inner_hit_prev = inner_hit_mid;
            veto_hit_prev = veto_hit_mid;
            n9_prev = n9_mid;
            n100_prev = n100_mid;
            n400_prev = n400_mid;
            nX_prev = nX_mid;
            closestPMT_prev = closestPMT_mid;
            inner_hit_prev = inner_hit_mid;
            veto_hit_prev = veto_hit_mid;
            //
            bonsai_goodness_prev = bonsai_goodness_mid;
            dir_goodness_prev = dir_goodness_mid;
            azi_ks_prev = azi_ks_mid;
            //
            xQFit_prev = xQFit_mid;
            yQFit_prev = yQFit_mid;
            zQFit_prev = zQFit_mid;
            closestPMTQFit_prev = closestPMTQFit_mid;

          
	    mid_x = next_x;
            mid_y = next_y;
            mid_z = next_z;
            mid_u = next_u;
            mid_v = next_v;
            mid_w = next_w;
            //
            mid_mcx = next_mcx;
            mid_mcy = next_mcy;
            mid_mcz = next_mcz;
            mid_mcu = next_mcu;
            mid_mcv = next_mcv;
            mid_mcw = next_mcw;
            mid_mct = next_mct;
            mc_energy_mid = mc_energy_next;
            //
	    timestamp_mid = timestamp_next;
            inner_hit_mid = inner_hit_next;
            veto_hit_mid = veto_hit_next;
            n9_mid = n9_next;
            n100_mid = n100_next;
            n400_mid = n400_next;
            nX_mid = nX_next;
            closestPMT_mid = closestPMT_next;
            inner_hit_mid = inner_hit_next;
            veto_hit_mid = veto_hit_next;
            //
            bonsai_goodness_mid = bonsai_goodness_next;
            dir_goodness_mid = dir_goodness_next;
            azi_ks_mid = azi_ks_next;
            //
            xQFit_mid = xQFit_next;
            yQFit_mid = yQFit_next;
            zQFit_mid = zQFit_next;
            closestPMTQFit_mid = closestPMTQFit_next;
         

            next_x = x;
            next_y = y;
            next_z = z;
            next_u = u;
            next_v = v;
            next_w = w;
            //
            next_mcx = mcx;
            next_mcy = mcy;
            next_mcz = mcz;
            next_mcu = mcu;
            next_mcv = mcv;
            next_mcw = mcw;
            next_mct = mct;
            mc_energy_next = mc_energy;
            //
            timestamp_next = timestamp;
            inner_hit_next = inner_hit;
            veto_hit_next = veto_hit;
            n9_next = n9;
            n100_next = n100;
            n400_next = n400;
            nX_next = nX;
            closestPMT_next = closestPMT;
            inner_hit_next = inner_hit;
            veto_hit_next = veto_hit;
            //
            bonsai_goodness_next = bonsai_goodness;
            dir_goodness_next = dir_goodness;
            azi_ks_next = azi_ks;
            //
            xQFit_next = xQFit;
            yQFit_next = yQFit;
            zQFit_next = zQFit;
            closestPMTQFit_next = closestPMTQFit;

 
          }
          QFit  = 0 ;
          xQFit = yQFit = zQFit = -999999.0;

	      beta_one_prev = beta_one;
          beta_two_prev = beta_two;
          beta_three_prev = beta_three;
          beta_four_prev = beta_four;
          beta_five_prev = beta_five;
          beta_six_prev = beta_six;

          // delete BONSAI objects and reset likelihoods
	  delete bsgrid;
	  delete bsgdn;
	  bslike->set_hits(NULL);
	} 
    }
  out->cd();
  data->Write();
  run_summary->Fill();
  run_summary->Write();
  out->Close();
  delete(bsfit);
  delete(bslike);
  delete(bsgeom);
  return 0;
}

int nwin(RAT::DS::PMTInfo *pmtinfo,
         float twin,float *v,int nfit,int *cfit,float *tfit,int *cwin)
{
    if (nfit<=0) return(0);

    float ttof[nfit],tsort[nfit],dx,dy,dz;
    int   hit,nwin=0,nwindow,hstart_test,hstart,hstop;

    // calculate t-tof for each hit
    for(hit=0; hit<nfit; hit++)
    {
        TVector3 pos=pmtinfo->GetPosition(cfit[hit]-1);
        dx=pos.X()*0.1-v[0];
        dy=pos.Y()*0.1-v[1];
        dz=pos.Z()*0.1-v[2];
        tsort[hit]=ttof[hit]=tfit[hit]-sqrt(dx*dx+dy*dy+dz*dz)*CM_TO_NS;
    }
    sort(tsort,tsort+nfit);

    // find the largest number of hits in a time window <= twin
    nwindow=1;
    hstart_test=hstart=0;
    while(hstart_test<nfit-nwindow)
    {
        hstop=hstart_test+nwindow;
        while((hstop<nfit) && (tsort[hstop]-tsort[hstart_test]<=twin))
        {
            hstart=hstart_test;
            nwindow++;
            hstop++;
        }
        hstart_test++;
    }
    hstop=hstart+nwindow-1;
    for(hit=0; hit<nfit; hit++)
    {
        if (ttof[hit]<tsort[hstart]) continue;
        if (ttof[hit]>tsort[hstop]) continue;
        cwin[nwin++]=cfit[hit];
    }
    if (nwin!=nwindow) printf("nwin error %d!=%d\n",nwin,nwindow);
    return(nwindow);
}

