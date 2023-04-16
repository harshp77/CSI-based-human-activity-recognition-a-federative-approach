## CSI based Human Activity Recognition : A FederativeApproach




### Introduction 

• Falls are the second leading cause of accidental or unintentional injury and deaths worldwide in elderly. ( WHO 17’ )

• Buildings are responsible for 28% of global energy consumption and 32% of global **CO2 emissions**.(IEA 22’)

### The Problem

❏ Traditional methods of ensors or cameras is require significant in investments.

❏ Data can not be moved (homes) to a central location due to privacy issues.




### The Solution

❏ **Non-intrusive** and accurate human activity recognition. Particularly beneficial in healthcare, security, and smart home environments where privacy is a concern.

❏ **Privacy preserving** federated learning allows data to be processed locally on individual devices.


### Major Contributions

★ **Auto-adaptive** federated learning architecture for Human Activity Recognition.

★ Channel state information : This approach is **non-invasive** and does not require any additional hardware, making it highly cost-effective

★ Incorporated **Bidirectional LSTM** for capturing time varying CSI data as a Local Model .

★ **Real-time** activity recognition : Our solution enables immediate feedback and response in scenarios such as healthcare or security




### Phase I (Dataset)

● Name: Bespoke homegrown data (Public)

● Environment : in an indoor office area where the Tx and Rx are located 3 m apart

● Frequency : Label includes 6 persons, 6 activities, denoted as “Lie down, Fall, Walk, Run, Sit down, Stand up,”

● Values : Consist of 45 phase columns and 45 amplitude columns


### Phase 2 (CSI feature Extraction)**

● PCA : Applied on CSI, yielded reduced dimension of CSI Data (Phase + Amplitude)

● STFT : On application yielded N number of frequency component

● Initial Selection : Selected top 25 component as most of the energy of activities is in lower frequencies

● Light-GBM : Yielded Feature ranking among frequency component

### Phase 3 (Federated Learning)

● Input/Output : 2D time-varying data in x and y axis , order of 500 by 90, and outs 7 classes

● Model : Each Local Instance consist of keras Sequential Bidirectional LSTM layers.

● Global Aggregation : Scalar multiplication of data cardinality.

● Runtime : Each prediction takes 880 ms.




### Phase 4 (Deployment)**

● Part 1 Router setup :

  ○ A Router with active internet connection

● Part 2 Server Setup :

  ○ A Local Node capable of capturing CSI Information

● Part 3 Model Setup :

  ○ Preferable an edge device like Raspberry Pi or arduino with signal sensor


## Experimental Results

### GCP Instance
![image](https://user-images.githubusercontent.com/76607486/232314346-165bfbf6-b2df-4c83-ab54-90d601db30c8.png)

### Carbon Tracker
![image](https://user-images.githubusercontent.com/76607486/232314464-70d4638c-7ab7-465c-978f-a44b77b638ac.png)


### Loss Plot 
![image](https://user-images.githubusercontent.com/76607486/232314452-a2f6b9c5-0079-4564-87c8-bdcce38ca31b.png)


### Confusion Matrix
![image](https://user-images.githubusercontent.com/76607486/232314499-9d51b864-e71c-4460-924b-f778ed7504c1.png)
