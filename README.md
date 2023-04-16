<a name="br1"></a>..

` `**CSI based Human Activity
Recognition : A Federative
 Approach**

**Industrial Academia Meet 2023**

Presented by: Harsh Pandey

International Institute of Information Technology, Naya Raipur




<a name="br2"></a>**Introduction**

• Falls are the second leading cause of

accidental or unintentional injury

**deaths** worldwide in elderly. ( *WHO* 17’ *)*

• Buildings are responsible for 28% of

global energy consumption and 32%

of global **CO2 emissions**.(IEA 22’)

\* International Institute of Information Technology, Naya Raipur 2




<a name="br3"></a>**The Problem**

❏ Traditional methods of

sensors or cameras is

require significant in

investments.

❏ Data can not be moved

(homes) to a central loca

**privacy issues**.

\* International Institute of Information Technology, Naya Raipur 3




<a name="br4"></a>**The Solution**

❏ **Non-intrusive** and accurate human activity

recognition. Particularly beneficial in healthcare,

security, and smart home environments where

privacy is a concern.

❏ **Privacy preserving** federated learning allows

data to be processed locally on individual devices.

\* International Institute of Information Technology, Naya Raipur 4



<a name="br5"></a>**Major Contributions**

★ **Auto-adaptive** federated learning architecture for Human

Activity Recognition.

★ Channel state information : This approach is **non-invasive**

and does not require any additional hardware, making it highly

cost-effective

★ Incorporated **Bidirectional LSTM** for capturing time varying

CSI data as a Local Model .

★ **Real-time** activity recognition : Our solution enables

immediate feedback and response in scenarios such as

\* International Institute of Information Technology, Naya Raipur 5

healthcare or security




<a name="br6"></a>**Proposed Architecture**

\* International Institute of Information Technology, Naya Raipur 6




<a name="br7"></a>**Solution Architecture**

` `**Figure 1 -- Figure 2 -- Federated
Local/Centralized CSI Architecture**

**Classification**

\* International Institute of Information Technology, Naya Raipur 7




<a name="br8"></a>**Phase I (Dataset)**

● **Name** : Bespoke homegrown data (Public)

● **Environment** : in an indoor office area

where the Tx and Rx are located 3 m

apart

● **Frequency** : Label includes 6 persons, 6

activities, denoted as “Lie down, Fall,

Walk, Run, Sit down, Stand up,”

● **Values :** Consist of 45 phase columns

and 45 amplitude columns

\* International Institute of Information Technology, Naya Raipur 8




<a name="br9"></a>**Phase 2 (CSI feature Extraction)**

● **PCA** : Applied on CSI, yielded reduced

dimension of CSI Data (Phase +

Amplitude)

● **STFT** : On application yielded N number of

frequency component

● **Initial Selection :** Selected top 25

component as most of the energy of

activities is in lower frequencies

● **Light-GBM** : Yielded Feature ranking

||<p>among frequency component</p><p>International Institute of Information Technology, Naya Raipur</p>||9|
| :- | :- | :- | :- |




<a name="br10"></a>**Phase 3 (Federated Learning)**

● **Input/Output** : 2D time-varying data in x

and y axis , order of 500 by 90, and outs

7 classes

● **Model :** Each Local Instance consist of

keras Sequential Bidirectional LSTM

layers.

● **Global Aggregation :** Scalar

multiplication of data cardinality.

● **Runtime :** Each prediction takes 880 ms.

Source: SpringerLink Article

Making it Real Time.

\* International Institute of Information Technology, Naya Raipur 10



<a name="br11"></a>**Phase 4 (Deployment)**

● **Part 1 Router setup :**

○ A Router with active internet

connection

● **Part 2 Server Setup :**

○ A Local Node capable of capturing CSI

Information

● **Part 3 Model Setup :**

○ Preferable an edge device like

Raspberry Pi or arduino with signal

sensor

\* International Institute of Information Technology, Naya Raipur 11



<a name="br12"></a>**Experimental Results**

||<p>**A) Centralized**</p><p>` `**learning**</p>||<p>**B) Federated**</p><p>` `**learning**</p>|
| :- | :- | :- | :- |
**C) Energy**

**Consumption D) Confusion
 Matrix**

\* International Institute of Information Technology, Naya Raipur 12




<a name="br13"></a>**Impact of our solution**

||<p>**Public Health**</p><p>Prevent falls and enable</p><p>remote monitoring of</p><p>patient.</p>||<p>**Privacy**</p><p>**Compliant**</p><p>Ensures the privacy and</p><p>security of user data as</p><p>the data never leaves</p><p>the user's pc</p>|
| :- | :- | :- | :- |
||<p>**Energy**</p><p>**Management**</p><p>Cost savings: In smart</p><p>homes, help automate</p><p>various tasks and reduce</p><p>energy consumption</p>||<p>**Self Learning**</p><p>Auto adapts to new</p><p>data no human</p><p>intervention required for</p><p>nodes addition.</p>|
13




<a name="br14"></a>**Future Scope**

||<p>**Context-aware**</p><p>**activity recognition**</p><p>Adaptation to diverse</p><p>environments and scenarios</p><p>improves human activity</p><p>recognition efficiency and</p><p>personalization.</p>||<p>**Transfer learning**</p><p>By leveraging pre-trained</p><p>models, transfer learning can</p><p>help reduce the amount of</p><p>labeled data required to</p><p>achieve high accuracy.</p>|
| :- | :- | :- | :- |
||<p>**Integration with other**</p><p>**technologies**</p>||**Edge computing**|
||<p>Combining it with other modalities,</p><p>such as audio or video, can lead to</p><p>more accurate and robust activity</p><p>recognition.</p>||<p>Edge deployment lowers latency</p><p>and enhances data privacy.</p>|
14




<a name="br15"></a>..

**Thank You**

\* International Institute of Information Technology, Naya Raipur 15
