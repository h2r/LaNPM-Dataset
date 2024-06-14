# LaNMP

LaNMP is a mobile manipulation robot dataset comprised of Natural Language, Navigation, Manipulation, and Perception (LaNMP) data. The dataset is collected in both simulated and real-world environments. The environments are multi-room, ensuring the tasks are long-horizon in nature. The tasks are pick-and-place described by humans to a robot in natural language. The trajectories, which are collected from robots via human teleoperation, contain LaNMP data at every timestep. There are 524 simulated and 50 real trajectories, totalling to 574 trajectories.

#### Dataset Link
<!-- info: Provide a link to the dataset: -->
<!-- width: half -->
https://www.dropbox.com/scl/fo/c1q9s420pzu1285t1wcud/AGMDPvgD5R1ilUFId0i94KE?rlkey=7lwmxnjagi7k9kgimd4v7fwaq&dl=0

#### Data Card Author(s)
<!-- info: Select **one role per** Data Card Author:

(Usage Note: Select the most appropriate choice to describe the author's role
in creating the Data Card.) -->
<!-- width: half -->
- **Name, Team:** Ahmed Jaafar (Owner)

## Authorship
### Publishers
#### Publishing Organization(s)
<!-- scope: telescope -->
<!-- info: Provide the names of the institution or organization responsible
for publishing the dataset: -->
Brown University, Rutgers University, University of Pennsylvania

#### Industry Type(s)
<!-- scope: periscope -->
<!-- info: Select **all applicable** industry types to which the publishing
organizations belong: -->
<!-- - Corporate - Tech -->
<!-- - Corporate - Non-Tech (please specify) -->
- Academic - Tech
<!-- - Academic - Non-Tech (please specify) -->
<!-- - Not-for-profit - Tech
- Not-for-profit - Non-Tech (please specify)
- Individual (please specify)
- Others (please specify) -->

#### Contact Detail(s)
<!-- scope: microscope -->
<!-- info: Provide publisher contact details: -->
- **Publishing POC:** Ahmed Jaafar
- **Affiliation:** Brown University
- **Contact:** ahmed_jaafar@brown.edu
- **Website:** https://lanmpdataset.github.io/

#### Author(s)
<!-- scope: microscope -->
<!-- info: Provide the details of all authors associated with the dataset:

(Usage Note: Provide the affiliation and year if different from publishing
institutions or multiple affiliations.) -->
- Ahmed Jaafar, Brown University
- Shreyas Sundara Raman, Brown University
- Yichen Wei, Brown University
- Sofia Juliani, Rutgers University
- Anneke Wernerfelt, University of Pennsylvania
- Ifrah Idrees, Brown University
- Jason Xinyu Liu, Brown University
- Stefanie Tellex, Associate Professor, Brown University

### Funding Sources
Coming soon!

<!-- #### Institution(s) -->
<!-- scope: telescope -->
<!-- info: Provide the names of the funding institution(s): -->
<!-- - Name of Institution
- Name of Institution
- Name of Institution

#### Funding or Grant Summary(ies)
<!-- scope: periscope -->
<!-- width: full -->
<!-- info: Provide a short summary of programs or projects that may have funded
the creation, collection, or curation of the dataset.

Use additional notes to capture any other relevant information or
considerations. -->
<!-- *For example, Institution 1 and institution 2 jointly funded this dataset as a
part of the XYZ data program, funded by XYZ grant awarded by institution 3 for
the years YYYY-YYYY.*

Summarize here. Link to documents if available.

**Additional Notes:** Add here -->

## Dataset Overview
#### Data Subject(s)
<!-- scope: telescope -->
<!-- info: Select ***all applicable**** subjects contained the dataset: -->
- Data about places and objects
- Synthetically generated data
- Data about systems or products and their behaviors
- Others (Language data provided by humans, robot movement and visual data)

#### Dataset Snapshot
<!-- scope: periscope -->
<!-- info: Provide a snapshot of the dataset:<br><br>(Use the additional notes
to include relevant information, considerations, and links to table(s) with
more detailed breakdowns.) -->
Category | Data
--- | ---
Size of Dataset | 288400 MB
Number of Instances | 574
Human Labels | 574
Capabilities | 4
Avg. Trajectory Length | 247
Number of environments | 8
Number of rooms | 30
Number of robots | 2


**Above:** The numbers are combining both the simulated and real datasets. "Capabilities" refers to the high-level aspects/modalities this dataset covers: Natural language, Navigation, Manipulation, and Perception. "Human Labels" refers to the natural language commands of robot tasks provided by humans.

<!-- **Additional Notes:** -->


#### Content Description
<!-- scope: microscope -->
<!-- info: Provide a short description of the content in a data point: -->
Every data point in simulation (trajectory time step) contains these important aspects: natural language command, egocentric RGB-D, instance segmentations, bounding boxes, robot body pose, robot end-effector pose, and grasped object poses.

Every data point in real (trajectory time step) contains on a high-level: natural language command, egocentric RGB-D, egocentric RGB-D, gripper RGB-D, gripper instance segmentations, robot body pose, robot arm pose, feet positions, joint angles, robot body velocity, robot arm velocity, gripper open percentage, object held boolean.

#### Descriptive Statistics
<!-- width: full -->
<!-- info: Provide basic descriptive statistics for each field.

Use additional notes to capture any other relevant information or
considerations.

Usage Note: Some statistics will be relevant for numeric data, for not for
strings. -->

Statistic | Field Name | Field Name | Field Name | Field Name | Field Name | Field Name
--- | --- | --- | --- | --- | --- | ---
count |
mean |
std |
min |
25% |
50% |
75% |
max |
mode |

**Above:** Provide a caption for the above table or visualization.

**Additional Notes:** Add here.

### Sensitivity of Data
#### Sensitivity Type(s)
<!-- scope: telescope -->
<!-- info: Select ***all applicable*** data types present in the dataset: -->
- User Content
- Anonymous Data
- Others (Robot movement and visual data)


#### Risk Type(s)
<!-- scope: telescope -->
<!-- info: Select **all applicable** risk types presenting from the
dataset: -->
- No Known Risks


### Dataset Version and Maintenance
#### Maintenance Status
<!-- scope: telescope -->
<!-- info: Select **one:** -->

**Actively Maintained** - No new versions will be made
available, but this dataset will
be actively maintained,
including but not limited to
updates to the data.


#### Version Details
<!-- scope: periscope -->
<!-- info: Provide details about **this** version of the dataset: -->
**Current Version:** 1.0

**Last Updated:** 06/2024

**Release Date:** 06/2024

#### Maintenance Plan
<!-- scope: microscope -->
<!-- info: Summarize the maintenance plan for the dataset:

Use additional notes to capture any other relevant information or
considerations. -->
Ahmed Jaafar will be mainting this dataset and resolving dataset issues brought up by the community.

<!-- **Versioning:** Summarize here. Include information about criteria for
versioning the dataset.

**Updates:** Summarize here. Include information about criteria for refreshing
or updating the dataset.

**Errors:** Summarize here. Include information about criteria for refreshing
or updating the dataset.

**Feedback:** Summarize here. Include information about criteria for refreshing
or updating the dataset.

**Additional Notes:** Add here -->

<!-- #### Next Planned Update(s)
<!-- scope: periscope -->
<!-- info: Provide details about the next planned update: -->
<!-- **Version affected:** 1.0

**Next data update:** MM/YYYY

**Next version:** 1.1

**Next version update:** MM/YYYY --> -->

<!-- #### Expected Change(s)
<!-- scope: microscope -->
<!-- info: Summarize the updates to the dataset and/or data that are expected
on the next update.

Use additional notes to capture any other relevant information or
considerations. -->
<!-- **Updates to Data:** Summarize here. Include links, charts, and visualizations
as appropriate.

**Updates to Dataset:** Summarize here. Include links, charts, and
visualizations as appropriate.

**Additional Notes:** Add here --> -->

## Example of Data Points
#### Primary Data Modality
<!-- scope: telescope -->
<!-- info: Select **one**: -->
- Image Data
- Text Data
- Tabular Data
- Audio Data
- Video Data
- Time Series
- Graph Data
- Geospatial Data
- Multimodel (please specify)
- Unknown
- Others (please specify)

#### Sampling of Data Points
<!-- scope: periscope -->
<!-- info: Provide link(s) to data points or exploratory demos: -->
- Demo Link
- Typical Data Point Link
- Outlier Data Point Link
- Other Data Point Link
- Other Data Point Link

#### Data Fields
<!-- scope: microscope -->
<!-- info: List the fields in data points and their descriptions.

(Usage Note: Describe each field in a data point. Optionally use this to show
the example.) -->

Field Name | Field Value | Description
--- | --- | ---
Field Name | Field Value | Description
Field Name | Field Value | Description
Field Name | Field Value | Description

**Above:** Provide a caption for the above table or visualization if used.

**Additional Notes:** Add here

#### Typical Data Point
<!-- width: half -->
<!-- info: Provide an example of a typical data point and describe what makes
it typical.

**Use additional notes to capture any other relevant information or
considerations.** -->
Summarize here. Include any criteria for typicality of data point.

```
{'q_id': '8houtx',
  'title': 'Why does water heated to room temperature feel colder than the air around it?',
  'selftext': '',
  'document': '',
  'subreddit': 'explainlikeimfive',
  'answers': {'a_id': ['dylcnfk', 'dylcj49'],
  'text': ["Water transfers heat more efficiently than air. When something feels cold it's because heat is being transferred from your skin to whatever you're touching. ... Get out of the water and have a breeze blow on you while you're wet, all of the water starts evaporating, pulling even more heat from you."],
  'score': [5, 2]},
  'title_urls': {'url': []},
  'selftext_urls': {'url': []},
  'answers_urls': {'url': []}}
```

**Additional Notes:** Add here


## Motivations & Intentions
### Motivations
#### Purpose(s)
<!-- scope: telescope -->
<!-- info: Select **one**: -->
- Research

#### Domain(s) of Application
<!-- scope: periscope -->
<!-- info: Provide a list of key domains of application that the dataset has
been designed for:<br><br>(Usage Note: Use comma-separated keywords.) -->

`Robotics`, `Imitation Learning`, `Reinfocement Learning`, `Machine Learning`

#### Motivating Factor(s)
<!-- scope: microscope -->
<!-- info: List the primary motivations for creating or curating this dataset:

(Usage Note: use this to describe the problem space and corresponding
motivations for the dataset.) -->
For example:

- Bringing demographic diversity to imagery training data for object-detection models
- Encouraging academics to take on second-order challenges of cultural representation in object detection

Summarize motivation here. Include links where relevant.

### Intended Use
#### Dataset Use(s)
<!-- scope: telescope -->
<!-- info: Select **one**: -->
- Safe for research use

#### Suitable Use Case(s)
<!-- scope: periscope -->
<!-- info: Summarize known suitable and intended use cases of this dataset.

Use additional notes to capture any specific patterns that readers should
look out for, or other relevant information or considerations. -->
**Suitable Use Case:** Summarize here. Include links where necessary.

**Suitable Use Case:** Summarize here. Include links where necessary.

**Suitable Use Case:** Summarize here. Include links where necessary.

**Additional Notes:** Add here


#### Research and Problem Space(s)
<!-- scope: periscope -->
<!-- info: Provide a description of the specific problem space that this
dataset intends to address. -->
Summarize here. Include any specific research questions.

#### Citation Guidelines
<!-- scope: microscope -->
<!-- info: Provide guidelines and steps for citing this dataset in research
and/or production.

Use additional notes to capture any specific patterns that readers should look
out for, or other relevant information or considerations. -->
**Guidelines & Steps:** Summarize here. Include links where necessary.

**BiBTeX:**
```

```

**Additional Notes:** Add here

## Access, Rentention, & Wipeout
### Access
#### Access Type
<!-- scope: telescope -->
<!-- info: Select **one**: -->
- External - Open Access

#### Documentation Link(s)
<!-- scope: periscope -->
<!-- info: Provide links that describe documentation to access this
dataset: -->
- Dataset Website URL: https://www.dropbox.com/scl/fo/c1q9s420pzu1285t1wcud/AGMDPvgD5R1ilUFId0i94KE?rlkey=7lwmxnjagi7k9kgimd4v7fwaq&dl=0
- GitHub URL: https://github.com/h2r/LaNPM-Dataset/

#### Prerequisite(s)
<!-- scope: microscope -->
<!-- info: Please describe any required training or prerequisites to access
this dataset. -->
For example:

This dataset requires membership in [specific] database groups:

- Complete the [Mandatory Training]
- Read [Data Usage Policy]
- Initiate a Data Requesting by filing

### Retention
#### Duration
<!-- scope: periscope -->
<!-- info: Specify the duration for which this dataset can be retained: -->
Specify duration in days, months, or years.

#### Policy Summary
<!-- scope: microscope -->
<!-- info: Summarize the retention policy for this dataset. -->
**Retention Plan ID:** Write here

**Summary:** Write summary and notes here

#### Process Guide
<!-- scope: periscope -->
<!-- info: Summarize any requirements and related steps to retain the dataset.

Use additional notes to capture any other relevant information or
considerations. -->
For example:

This dataset compiles with [standard policy guidelines].

**Additional Notes:** Add here

#### Exception(s) and Exemption(s)
<!-- scope: microscope -->
<!-- info: Summarize any exceptions and related steps to retain the dataset.
Include links where necessary.

Use additional notes to capture any other relevant information or
considerations. -->
**Exemption Code:** `ANONYMOUS_DATA` /
`EMPLOYEE_DATA` / `PUBLIC_DATA` /
`INTERNAL_BUSINESS_DATA` /
`SIMULATED_TEST_DATA`

**Summary:** Write summary and notes here.

**Additional Notes:** Add here

### Wipeout and Deletion
#### Duration
<!-- scope: periscope -->
<!-- info: Specify the duration after which this dataset should be deleted or
wiped out: -->
Specify duration in days, months, or years.

#### Deletion Event Summary
<!-- scope: microscope -->
<!-- info: Summarize the sequence of events and allowable processing for data
deletion.

Use additional notes to capture any other relevant information or
considerations. -->
**Sequence of deletion and processing events:**

- Summarize first event here
- Summarize second event here
- Summarize third event here

**Additional Notes:** Add here

#### Acceptable Means of Deletion
<!-- scope: periscope -->
<!-- info: List the acceptable means of deletion: -->
- Write acceptable means of deletion
- Write acceptable means of deletion
- Write acceptable means of deletion

#### Post-Deletion Obligations
<!-- scope: microscope -->
<!-- info: Summarize the sequence of obligations after a deletion event.

**Use additional notes to capture any other relevant information or
considerations.** -->
**Sequence of post-deletion obligations:**

- Summarize first obligation here
- Summarize second obligation here
- Summarize third obligation here

**Additional Notes:** Add here

#### Operational Requirement(s)
<!-- scope: periscope -->
<!-- info: List any wipeout integration operational requirements: -->
**Wipeout Integration Operational Requirements:**

- Write first requirement here
- Write second requirement here
- Write third requirement here

#### Exceptions and Exemptions
<!-- scope: microscope -->
<!-- info: Summarize any exceptions and related steps to a deletion event.

**Use additional notes to capture any other relevant information or
considerations.** -->
**Policy Exception bug:** [bug]

**Summary:** Write summary and notes here

**Additional Notes:** Add here

## Provenance
### Collection
#### Method(s) Used
<!-- scope: telescope -->
<!-- info: Select **all applicable** methods used to collect data: -->
- API
- Artificially Generated
- Crowdsourced - Paid
- Crowdsourced - Volunteer
- Vendor Collection Efforts
- Scraped or Crawled
- Survey, forms, or polls
- Taken from other existing datasets
- Unknown
- To be determined
- Others (please specify)

#### Methodology Detail(s)
<!-- scope: periscope -->
<!-- info: Provide a description of each collection method used.

Use additional notes to capture any other relevant information or
considerations.

(Usage Note: Duplicate and complete the following for collection method
type.) -->
**Collection Type**

**Source:** Describe here. Include links where available.

**Platform:** [Platform Name], Describe platform here. Include links where relevant.

**Is this source considered sensitive or high-risk?** [Yes/No]

**Dates of Collection:** [MMM YYYY - MMM YYYY]

**Primary modality of collection data:**

*Usage Note: Select one for this collection type.*

- Image Data
- Text Data
- Tabular Data
- Audio Data
- Video Data
- Time Series
- Graph Data
- Geospatial Data
- Unknown
- Multimodal (please specify)
- Others (please specify)

**Update Frequency for collected data:**

*Usage Note: Select one for this collection type.*

- Yearly
- Quarterly
- Monthly
- Biweekly
- Weekly
- Daily
- Hourly
- Static
- Others (please specify)

**Additional Links for this collection:**

- [Access Policy]
- [Wipeout Policy]
- [Retention Policy]

**Additional Notes:** Add here

#### Source Description(s)
<!-- scope: microscope -->
<!-- info: Provide a description of each upstream source of data.

Use additional notes to capture any other relevant information or
considerations. -->
- **Source:** Describe here. Include links, data examples, metrics, visualizations where relevant.
- **Source:** Describe here. Include links, data examples, metrics, visualizations where relevant.
- **Source:** Describe here. Include links, data examples, metrics, visualizations where relevant.

**Additional Notes:** Add here

#### Collection Cadence
<!-- scope: telescope -->
<!-- info: Select **all applicable**: -->
**Static:** Data was collected once from single or multiple sources.

**Streamed:** Data is continuously acquired from single or multiple sources.

**Dynamic:** Data is updated regularly from single or multiple sources.

**Others:** Please specify

#### Data Integration
<!-- scope: periscope -->
<!-- info: List all fields collected from different sources, and specify if
they were included or excluded from the dataset.

Use additional notes to
capture any other relevant information or considerations.

(Usage Note: Duplicate and complete the following for each upstream
source.) -->
**Source**

**Included Fields**

Data fields that were collected and are included in the dataset.

Field Name | Description
--- | ---
Field Name | Describe here. Include links, data examples, metrics, visualizations where relevant.
Field Name | Describe here. Include links, data examples, metrics, visualizations where relevant.

**Additional Notes:** Add here

**Excluded Fields**

Data fields that were collected but are excluded from the dataset.

Field Name | Description
--- | ---
Field Name | Describe here. Include links, data examples, metrics, visualizations where relevant.
Field Name | Describe here. Include links, data examples, metrics, visualizations where relevant.

**Additional Notes:** Add here

#### Data Processing
<!-- scope: microscope -->
<!-- info: Summarize how data from different sources or methods aggregated,
processed, or connected.

Use additional notes to capture any other
relevant information or considerations.

(Usage Note: Duplicate and complete the following for each source OR
collection method.) -->
**Collection Method or Source**

**Description:** Describe here. Include links where relevant.

**Methods employed:** Describe here. Include links where relevant.

**Tools or libraries:** Describe here. Include links where relevant.

**Additional Notes:** Add here

### Collection Criteria
#### Data Selection
<!-- scope: telescope -->
<!-- info: Summarize the data selection criteria.

Use additional notes to capture any other relevant information or
considerations. -->
- **Collection Method of Source:** Summarize data selection criteria here. Include links where available.
- **Collection Method of Source:** Summarize data selection criteria here. Include links where available.
- **Collection Method of Source:** Summarize data selection criteria here. Include links where available.

**Additional Notes:** Add here

#### Data Inclusion
<!-- scope: periscope -->
<!-- info: Summarize the data inclusion criteria.

Use additional notes to capture any other relevant information or
considerations. -->
- **Collection Method of Source:** Summarize data inclusion criteria here. Include links where available.
- **Collection Method of Source:** Summarize data inclusion criteria here. Include links where available.
- **Collection Method of Source:** Summarize data inclusion criteria here. Include links where available.

**Additional Notes:** Add here

#### Data Exclusion
<!-- scope: microscope -->
<!-- info: Summarize the data exclusion criteria.

Use additional notes to capture any other relevant information or
considerations. -->
- **Collection Method of Source:** Summarize data exclusion criteria here. Include links where available.
- **Collection Method of Source:** Summarize data exclusion criteria here. Include links where available.
- **Collection Method of Source:** Summarize data exclusion criteria here. Include links where available.

**Additional Notes:** Add here

### Relationship to Source
#### Use & Utility(ies)
<!-- scope: telescope -->
<!-- info: Describe how the resulting dataset is aligned with the purposes,
motivations, or intended use of the upstream source(s).

Use additional notes to capture any other relevant information or
considerations.

(Usage Note: Duplicate and complete the following for each source type.) -->
- **Source Type:** Summarize here. Include links where available.
- **Source Type:** Summarize here. Include links where available.
- **Source Type:** Summarize here. Include links where available.

**Additional Notes:** Add here

#### Benefit and Value(s)
<!-- scope: periscope -->
<!-- info: Summarize the benefits of the resulting dataset to its consumers,
compared to the upstream source(s).

Use additional notes to capture any other relevant information or
considerations.

(Usage Note: Duplicate and complete the following for each source type.) -->
- **Source Type:** Summarize here. Include links where available.
- **Source Type:** Summarize here. Include links where available.
- **Source Type:** Summarize here. Include links where available.

**Additional Notes:** Add here

#### Limitation(s) and Trade-Off(s)
<!-- scope: microscope -->
<!-- info: What are the limitations of the resulting dataset to its consumers,
compared to the upstream source(s)?

Break down by source type.<br><br>(Usage Note: Duplicate and complete the
following for each source type.) -->
- **Source Type:** Summarize here. Include links where available.
- **Source Type:** Summarize here. Include links where available.
- **Source Type:** Summarize here. Include links where available.

### Version and Maintenance
<!-- info: Fill this next row if this is not the first version of the dataset,
and there is no data card available for the first version -->
#### First Version
<!-- scope: periscope -->
<!-- info: Provide a **basic description of the first version** of this
dataset. -->
- **Release date:** MM/YYYY
- **Link to dataset:** [Dataset Name + Version]
- **Status:** [Select one: Actively Maintained/Limited Maintenance/Deprecated]
- **Size of Dataset:** 123 MB
- **Number of Instances:** 123456

#### Note(s) and Caveat(s)
<!-- scope: microscope -->
<!-- info: Summarize the caveats or nuances of the first version of this
dataset that may affect the use of the current version.

Use additional notes to capture any other relevant information or
considerations. -->
Summarize here. Include links where available.

**Additional Notes:** Add here

#### Cadence
<!-- scope: telescope -->
<!-- info: Select **one**: -->
- Yearly
- Quarterly
- Monthly
- Biweekly
- Weekly
- Daily
- Hourly
- Static
- Others (please specify)

#### Last and Next Update(s)
<!-- scope: periscope -->
<!-- info: Please describe the update schedule: -->
- **Date of last update:** DD/MM/YYYY
- **Total data points affected:** 12345
- **Data points updated:** 12345
- **Data points added:** 12345
- **Data points removed:** 12345
- **Date of next update:** DD/MM/YYYY

#### Changes on Update(s)
<!-- scope: microscope -->
<!-- info: Summarize the changes that occur when the dataset is refreshed.

Use additional notes to capture any other relevant information or
considerations.

(Usage Note: Duplicate and complete the following for each source type.) -->
- **Source Type:** Summarize here. Include links where available.
- **Source Type:** Summarize here. Include links where available.
- **Source Type:** Summarize here. Include links where available.

**Additional Notes:** Add here

## Human and Other Sensitive Attributes
#### Sensitive Human Attribute(s)
<!-- scope: telescope -->
<!-- info: Select **all attributes** that are represented (directly or
indirectly) in the dataset. -->
- Gender
- Socio-economic status
- Geography
- Language
- Age
- Culture
- Experience or Seniority
- Others (please specify)

#### Intentionality
<!-- scope: periscope -->
<!-- info: List fields in the dataset that contain human attributes, and
specify if their collection was intentional or unintentional.

Use additional notes to capture any other relevant information or
considerations. -->
**Intentionally Collected Attributes**

Human attributes were labeled or collected as a part of the dataset creation
process.

Field Name | Description
--- | ---
Field Name | Human Attributed Collected
Field Name | Human Attributed Collected

**Additional Notes:** Add here

**Unintentionally Collected Attributes**

Human attributes were not explicitly collected as a part of the dataset
creation process but can be inferred using additional methods.

Field Name | Description
--- | ---
Field Name | Human Attributed Collected
Field Name | Human Attributed Collected

**Additional Notes:** Add here

#### Rationale
<!-- scope: microscope -->
<!-- info: Describe the motivation, rationale, considerations or approaches
that caused this dataset to include the indicated human attributes.

Summarize why or how this might affect the use of the dataset. -->
Summarize here. Include links, table, and media as relevant.

#### Source(s)
<!-- scope: periscope -->
<!-- info: List the sources of the human attributes.

Use additional notes to capture any other relevant information or
considerations. -->
- **Human Attribute:** Sources
- **Human Attribute:** Sources
- **Human Attribute:** Sources

**Additional Notes:** Add here

#### Methodology Detail(s)
<!-- scope: microscope -->
<!-- info: Describe the methods used to collect human attributes in the
dataset.

Use additional notes to capture any other relevant information or
considerations.

(Usage Note: Duplicate and complete the following for each human
attribute.) -->

**Human Attribute Method:** Describe the collection method here. Include links where necessary

**Collection task:** Describe the task here. Include links where necessary

**Platforms, tools, or libraries:**

- [Platform, tools, or libraries]: Write description here
- [Platform, tools, or libraries]: Write description here
- [Platform, tools, or libraries]: Write description here

**Additional Notes:** Add here

#### Distribution(s)
<!-- width: full -->
<!-- info: Provide basic descriptive statistics for each human attribute,
noting key takeaways in the caption.

Use additional notes to capture any other relevant information or
considerations.

(Usage Note: Duplicate and complete the following for each human
attribute.) -->
Human Attribute | Label or Class | Label or Class | Label or Class | Label or Class
--- | --- | --- | --- | ---
Count | 123456 | 123456 | 123456 | 123456
[Statistic] | 123456 | 123456 | 123456 | 123456
[Statistic] | 123456 | 123456 | 123456 | 123456
[Statistic] | 123456 | 123456 | 123456 | 123456

**Above:** Provide a caption for the above table or visualization.
**Additional Notes:** Add here

#### Known Correlations
<!-- scope: periscope -->
<!-- info: Describe any known correlations with the indicated sensitive
attributes in this dataset.

Use additional notes to capture any other relevant information or
considerations.

(Usage Note: Duplicate for each known correlation.) -->
[`field_name`, `field_name`]

**Description:** Summarize here. Include visualizations, metrics, or links
where necessary.

**Impact on dataset use:** Summarize here. Include visualizations, metrics, or
links where necessary.

**Additional Notes:** add here

#### Risk(s) and Mitigation(s)
<!-- scope: microscope -->
<!-- info: Summarize systemic or residual risks, performance expectations,
trade-offs and caveats because of human attributes in this dataset.

Use additional notes to capture any other relevant information or
considerations.

Usage Note: Duplicate and complete the following for each human attribute. -->
**Human Attribute**

Summarize here. Include links and metrics where applicable.

**Risk type:** [Description + Mitigations]

**Risk type:** [Description + Mitigations]

**Risk type:** [Description + Mitigations]

**Trade-offs, caveats, & other considerations:** Summarize here. Include
visualizations, metrics, or links where necessary.

**Additional Notes:** Add here

## Extended Use
### Use with Other Data
#### Safety Level
<!-- scope: telescope -->
<!-- info: Select **one**: -->
- Safe to use with other data
- Conditionally safe to use with other data
- Should not be used with other data
- Unknown
- Others (please specify)

#### Known Safe Dataset(s) or Data Type(s)
<!-- scope: periscope -->
<!-- info: List the known datasets or data types and corresponding
transformations that **are safe to join or aggregate** this dataset with. -->
**Dataset or Data Type:** Summarize here. Include visualizations, metrics,
or links where necessary.

**Dataset or Data Type:** Summarize here. Include visualizations, metrics,
or links where necessary.

**Dataset or Data Type:** Summarize here. Include visualizations, metrics,
or links where necessary.

#### Best Practices
<!-- scope: microscope -->
<!-- info: Summarize best practices for using this dataset with other datasets
or data types.

Use additional notes to capture any other relevant information or
considerations. -->
Summarize here. Include visualizations, metrics, demonstrative examples,
or links where necessary.

**Additional Notes:** Add here

#### Known Unsafe Dataset(s) or Data Type(s)
<!-- scope: periscope -->
<!-- info: Fill this out if you selected "Conditionally safe to use with other
datasets" or "Should not be used with other datasets":

List the known datasets or data types and corresponding transformations that
are **unsafe to join or aggregate** with this dataset. -->
**Dataset or Data Type:** Summarize here. Include visualizations, metrics,
or links where necessary.

**Dataset or Data Type:** Summarize here. Include visualizations, metrics,
or links where necessary.

**Dataset or Data Type:** Summarize here. Include visualizations, metrics,
or links where necessary.

#### Limitation(s) and Recommendation(s)
<!-- scope: microscope -->
<!-- info: Fill this out if you selected "Conditionally safe to use with
other datasets" or "Should not be used with
other datasets":

Summarize limitations of the dataset that introduce foreseeable risks when the
dataset is conjoined with other datasets.

Use additional notes to capture any other relevant information or
considerations. -->
Summarize here. Include links and metrics where applicable.

**Limitation type:** Dataset or data type, description and recommendation.

**Limitation type:** Dataset or data type, description and recommendation.

**Limitation type:** Dataset or data type, description and recommendation.

**Additional Notes:** Add here

### Forking & Sampling
#### Safety Level
<!-- scope: telescope -->
<!-- info: Select **one**: -->
- Safe to form and/or sample
- Conditionally safe to fork and/or sample
- Should not be forked and/or sampled
- Unknown
- Others (please specify)

#### Acceptable Sampling Method(s)
<!-- scope: periscope -->
<!-- info: Select **all applicable** acceptable methods to sample this
dataset: -->
- Cluster Sampling
- Haphazard Sampling
- Multi-stage sampling
- Random Sampling
- Retrospective Sampling
- Stratified Sampling
- Systematic Sampling
- Weighted Sampling
- Unknown
- Unsampled
- Others (please specify)

#### Best Practice(s)
<!-- scope: microscope -->
<!-- info: Summarize the best practices for forking or sampling this dataset.

Use additional notes to capture any other relevant information or
considerations. -->
Summarize here. Include links, figures, and demonstrative examples where
available.

**Additional Notes:** Add here

#### Risk(s) and Mitigation(s)
<!-- scope: periscope -->
<!-- info: Fill this out if you selected "Conditionally safe to fork and/or
sample" or "Should not be forked and/or sampled":

Summarize known or residual risks associated with forking and sampling methods
when applied to the dataset.

Use additional notes to capture any other
relevant information or considerations. -->
Summarize here. Include links and metrics where applicable.

**Risk Type:** [Description + Mitigations]

**Risk Type:** [Description + Mitigations]

**Risk Type:** [Description + Mitigations]

**Additional Notes:** Add here

#### Limitation(s) and Recommendation(s)
<!-- scope: microscope -->
<!-- info: Fill this out if you selected "Conditionally safe to fork and/or
sample" or "Should not be forked and/or sample":

Summarize the limitations that the dataset introduces when forking
or sampling the dataset and corresponding recommendations.

Use additional notes to capture any other relevant information or
considerations. -->
Summarize here. Include links and metrics where applicable.

**Limitation Type:** [Description + Recommendation]

**Limitation Type:** [Description + Recommendation]

**Limitation Type:** [Description + Recommendation]

**Additional Notes:** Add here

### Use in ML or AI Systems
#### Dataset Use(s)
<!-- scope: telescope -->
<!-- info: Select **all applicable** -->
- Training
- Testing
- Validation
- Fine Tuning

#### Notable Feature(s)
<!-- scope: periscope -->
<!-- info: Describe any notable feature distributions or relationships between
individual instances made explicit.

Include links to servers where readers can explore the data on their own. -->

**Exploration Demo:** [Link to server or demo.]

**Notable Field Name:** Describe here. Include links, data examples, metrics,
visualizations where relevant.

**Above:** Provide a caption for the above table or visualization.

**Additional Notes:** Add here

#### Usage Guideline(s)
<!-- scope: microscope -->
<!-- info: Summarize usage guidelines or policies that consumers should be
aware of.

Use additional notes to capture any other relevant information or
considerations. -->
**Usage Guidelines:** Summarize here. Include links where necessary.

**Approval Steps:** Summarize here. Include links where necessary.

**Reviewer:** Provide the name of a reviewer for publications referencing
this dataset.

**Additional Notes:** Add here

#### Distribution(s)
<!-- scope: periscope -->
<!-- info: Describe the recommended splits and corresponding criteria.

Use additional notes to capture any other
relevant information or considerations. -->

Set | Number of data points
--- | ---
Train | 62,563
Test | 62,563
Validation | 62,563
Dev | 62,563

**Above:** Provide a caption for the above table or visualization.

**Additional Notes:** Add here

#### Known Correlation(s)
<!-- scope: microscope -->
<!-- info: Summarize any known correlations with
the indicated features in this dataset.

Use additional notes to capture any other
relevant information or considerations.

(Usage Note: Duplicate for each known
correlation.) -->
`field_name`, `field_name`

**Description:** Summarize here. Include
visualizations, metrics, or links where
necessary.

**Impact on dataset use:** Summarize here.
Include visualizations, metrics, or links
where necessary.

**Risks from correlation:** Summarize here.
Include recommended mitigative steps if
available.

**Additional Notes:** Add here

#### Split Statistics
<!-- scope: periscope -->
<!-- width: full -->
<!-- info: Provide the sizes of each split. As appropriate, provide any
descriptive statistics for features. -->

Statistic | Train | Test | Valid | Dev
--- | --- | --- | --- | ---
Count | 123456 | 123456 | 123456 | 123456
Descriptive Statistic | 123456 | 123456 | 123456 | 123456
Descriptive Statistic | 123456 | 123456 | 123456 | 123456
Descriptive Statistic | 123456 | 123456 | 123456 | 123456

**Above:** Caption for table above.

## Transformations
<!-- info: Fill this section if any transformations were applied in the
creation of your dataset. -->
### Synopsis
#### Transformation(s) Applied
<!-- scope: telescope -->
<!-- info: Select **all applicable** transformations
that were applied to the dataset. -->
- Anomaly Detection
- Cleaning Mismatched Values
- Cleaning Missing Values
- Converting Data Types
- Data Aggregation
- Dimensionality Reduction
- Joining Input Sources
- Redaction or Anonymization
- Others (Please specify)

#### Field(s) Transformed
<!-- scope: periscope -->
<!-- info: Provide the fields in the dataset that
were transformed.

Use additional notes to capture any
other relevant information or
considerations.

(Usage Note: Duplicate and complete
the following for each transformation
type applied. Include the data types to
which fields were transformed.) -->
**Transformation Type**

Field Name | Source & Target
--- | ---
Field Name | Source Field: Target Field
Field Name | Source Field: Target Field
... | ...

**Additional Notes:** Add here

#### Library(ies) and Method(s) Used
<!-- scope: microscope -->
<!-- info: Provide a description of the methods
used to transform or process the
dataset.

Use additional notes to capture any
other relevant information or
considerations.

(Usage Note: Duplicate and complete
the following for each transformation
type applied.) -->
**Transformation Type**

**Method:** Describe the transformation
method here. Include links where
necessary.

**Platforms, tools, or libraries:**
- Platform, tool, or library: Write description here
- Platform, tool, or library: Write description here
- Platform, tool, or library: Write description here

**Transformation Results:** Provide
results, outcomes, and actions taken
because of the transformations. Include
visualizations where available.

**Additional Notes:** Add here

### Breakdown of Transformations
<!-- info: Fill out relevant rows. -->
#### Cleaning Missing Value(s)
<!-- scope: telescope -->
<!-- info: Which fields in the data were missing
values? How many? -->
Summarize here. Include links where available.

**Field Name:** Count or description

**Field Name:** Count or description

**Field Name:** Count or description

#### Method(s) Used
<!-- scope: periscope -->
<!-- info: How were missing values cleaned?
What other choices were considered? -->
Summarize here. Include links where necessary.

**Platforms, tools, or libraries**

- Platform, tool, or library: Write description here
- Platform, tool, or library: Write description here
- Platform, tool, or library: Write description here

#### Comparative Summary
<!-- scope: microscope -->
<!-- info: Why were missing values cleaned using
this method (over others)? Provide
comparative charts showing before
and after missing values were cleaned. -->
Summarize here. Include links, tables, visualizations where available.

**Field Name** | **Diff**
--- | ---
Field Name | Before: After
Field Name | Before: After
... | ...

**Above:** Provide a caption for the above table or visualization.

**Additional Notes:** Add here

#### Residual & Other Risk(s)
<!-- scope: telescope -->
<!-- info: What risks were introduced because of
this transformation? Which risks were
mitigated? -->
Summarize here. Include links and metrics where applicable.

- **Risk Type:** Description + Mitigations
- **Risk Type:** Description + Mitigations
- **Risk Type:** Description + Mitigations

#### Additional Considerations
<!-- scope: microscope -->
<!-- info: What additional considerations were
made? -->
Summarize here. Include links where available.

#### Cleaning Mismatched Value(s)
<!-- scope: telescope -->
<!-- info: Which fields in the data were corrected
for mismatched values? -->
Summarize here. Include links where available.

**Field Name:** Count or Description

**Field Name:** Count or Description

**Field Name:** Count or Description

#### Method(s) Used
<!-- scope: periscope -->
<!-- info: How were incorrect or mismatched
values cleaned? What other choices
were considered? -->
Summarize here. Include links where available.

#### Comparative Summary
<!-- scope: microscope -->
<!-- info: Why were incorrect or mismatched
values cleaned using this method (over
others)? Provide a comparative
analysis demonstrating before and
after values were cleaned. -->
Summarize here. Include links where available.

**Field Name** | **Diff**
--- | ---
Field Name | Before: After
Field Name | Before: After
... | ...

**Above:** Provide a caption for the above table or visualization.

**Additional Notes:** Add here

#### Residual & Other Risk(s)
<!-- scope: telescope -->
<!-- info: What risks were introduced because of
this transformation? Which risks were
mitigated? -->
Summarize here. Include links and metrics where applicable.

**Risk Type:** Description + Mitigations

**Risk Type:** Description + Mitigations

**Risk Type:** Description + Mitigations

#### Additional Considerations
<!-- scope: microscope -->
<!-- info: What additional considerations were made? -->
Summarize here. Include links where available.

#### Anomalies
<!-- scope: telescope -->
<!-- info: How many anomalies or outliers were
detected?
If at all, how were detected anomalies
or outliers handled?
Why or why not? -->
Summarize here. Include links where available.

**Field Name:** Count or Description

**Field Name:** Count or Description

**Field Name:** Count or Description

#### Method(s) Used
<!-- scope: periscope -->
<!-- info: What methods were used to detect
anomalies or outliers? -->
Summarize here. Include links where necessary.

**Platforms, tools, or libraries**

- Platform, tool, or library: Write description here
- Platform, tool, or library: Write description here
- Platform, tool, or library: Write description here

#### Comparative Summary
<!-- scope: microscope -->
<!-- info: Provide a comparative analysis
demonstrating before and after
anomaly handling measures. -->
Summarize here. Include links, tables, visualizations where available.

**Field Name** | **Diff**
--- | ---
Field Name | Before: After
Field Name | Before: After
... | ...

**Above:** Provide a caption for the above table or visualization.

**Additional Notes:** Add here

#### Residual & Other Risk(s)
<!-- scope: telescope -->
<!-- info: What risks were introduced because of
this transformation? Which risks were
mitigated? -->
Summarize here. Include links and metrics where applicable.

**Risk Type:** Description + Mitigations

**Risk Type:** Description + Mitigations

**Risk Type:** Description + Mitigations

#### Additional Considerations
<!-- scope: microscope -->
<!-- info: What additional considerations were made? -->
Summarize here. Include links where available.

#### Dimensionality Reduction
<!-- scope: telescope -->
<!-- info: How many original features were
collected and how many dimensions
were reduced? -->
Summarize here. Include links where available.

**Field Name:** Count or Description

**Field Name:** Count or Description

**Field Name:** Count or Description

#### Method(s) Used
<!-- scope: periscope -->
<!-- info: What methods were used to reduce the
dimensionality of the data? What other
choices were considered? -->
Summarize here. Include links where
necessary.

**Platforms, tools, or libraries**

- Platform, tool, or library: Write description here
- Platform, tool, or library: Write description here
- Platform, tool, or library: Write description here

#### Comparative Summary
<!-- scope: microscope -->
<!-- info: Why were features reduced using this
method (over others)? Provide
comparative charts showing before
and after dimensionality reduction
processes. -->
Summarize here. Include links, tables, visualizations where available.

**Field Name** | **Diff**
--- | ---
Field Name | Before: After
Field Name | Before: After
... | ...

**Above:** Provide a caption for the above table or visualization.

**Additional Notes:** Add here

#### Residual & Other Risks
<!-- scope: telescope -->
<!-- info: What risks were introduced because of
this transformation? Which risks were
mitigated? -->
Summarize here. Include links and metrics where applicable.

**Risk Type:** Description + Mitigations

**Risk Type:** Description + Mitigations

**Risk Type:** Description + Mitigations

#### Additional Considerations
<!-- scope: microscope -->
<!-- info: What additional considerations were made? -->
Summarize here. Include links where available.

#### Joining Input Sources
<!-- scope: telescope -->
<!-- info: What were the distinct input sources that were joined? -->
Summarize here. Include links where available.

**Field Name:** Count or Description

**Field Name:** Count or Description

**Field Name:** Count or Description

#### Method(s) Used
<!-- scope: periscope -->
<!-- info: What are the shared columns of fields used to join these
sources? -->
Summarize here. Include links where necessary.

**Platforms, tools, or libraries**

- Platform, tool, or library: Write description here
- Platform, tool, or library: Write description here
- Platform, tool, or library: Write description here

#### Comparative Summary
<!-- scope: microscope -->
<!-- info: Why were features joined using this
method over others?

Provide comparative charts showing
before and after dimensionality
reduction processes. -->
Summarize here. Include links, tables, visualizations where available.

**Field Name** | **Diff**
--- | ---
Field Name | Before: After
Field Name | Before: After
... | ...

**Above:** Provide a caption for the above table or visualization.

**Additional Notes:** Add here

#### Residual & Other Risk(s)
<!-- scope: telescope -->
<!-- info: What risks were introduced because of
this transformation? Which risks were
mitigated? -->
Summarize here. Include links and metrics where applicable.

**Risk Type:** Description + Mitigations

**Risk Type:** Description + Mitigations

**Risk Type:** Description + Mitigations


#### Additional Considerations
<!-- scope: microscope -->
<!-- info: What additional considerations were
made? -->
Summarize here. Include links where
available.

#### Redaction or Anonymization
<!-- scope: telescope -->
<!-- info: Which features were redacted or
anonymized? -->
Summarize here. Include links where available.

**Field Name:** Count or Description

**Field Name:** Count or Description

**Field Name:** Count or Description

#### Method(s) Used
<!-- scope: periscope -->
<!-- info: What methods were used to redact or
anonymize data? -->
Summarize here. Include links where necessary.

**Platforms, tools, or libraries**

- Platform, tool, or library: Write description here
- Platform, tool, or library: Write description here
- Platform, tool, or library: Write description here

#### Comparative Summary
<!-- scope: microscope -->
<!-- info: Why was data redacted or anonymized
using this method over others? Provide
comparative charts showing before
and after redaction or anonymization
process. -->
Summarize here. Include links, tables, visualizations where available.

**Field Name** | **Diff**
--- | ---
Field Name | Before: After
Field Name | Before: After
... | ...

**Above:** Provide a caption for the above table or visualization.

**Additional Notes:** Add here

#### Residual & Other Risk(s)
<!-- scope: telescope -->
<!-- info: What risks were introduced because of
this transformation? Which risks were
mitigated? -->
Summarize here. Include links and metrics where applicable.

**Risk Type:** Description + Mitigations

**Risk Type:** Description + Mitigations

**Risk Type:** Description + Mitigations


#### Additional Considerations
<!-- scope: microscope -->
<!-- info: What additional considerations were
made? -->
Summarize here. Include links where available.

#### Others (Please Specify)
<!-- scope: telescope -->
<!-- info: What was done? Which features or
fields were affected? -->
Summarize here. Include links where available.

**Field Name:** Count or Description

**Field Name:** Count or Description

**Field Name:** Count or Description

#### Method(s) Used
<!-- scope: periscope -->
<!-- info: What method were used? -->
Summarize here. Include links where necessary.

**Platforms, tools, or libraries**

- Platform, tool, or library: Write description here
- Platform, tool, or library: Write description here
- Platform, tool, or library: Write description here

#### Comparative Summary
<!-- scope: microscope -->
<!-- info: Why was this method used over
others? Provide comparative charts
showing before and after this
transformation. -->
Summarize here. Include links, tables, visualizations where available.

**Field Name** | **Diff**
--- | ---
Field Name | Before: After
Field Name | Before: After
... | ...

**Above:** Provide a caption for the above table or visualization.

**Additional Notes:** Add here

#### Residual & Other Risk(s)
<!-- scope: telescope -->
<!-- info: What risks were introduced because of
this transformation? Which risks were
mitigated? -->
Summarize here. Include links and metrics where applicable.

**Risk type:** [Description + Mitigations]

**Risk type:** [Description + Mitigations]

**Risk type:** [Description + Mitigations]

#### Additional Considerations
<!-- scope: microscope -->
<!-- info: What additional considerations were made? -->
Summarize here. Include links where available.

## Annotations & Labeling
<!-- info: Fill this section if any human or algorithmic annotation tasks were
performed in the creation of your dataset. -->
#### Annotation Workforce Type
<!-- scope: telescope -->
<!-- info: Select **all applicable** annotation
workforce types or methods used
to annotate the dataset: -->
- Annotation Target in Data
- Machine-Generated
- Annotations
- Human Annotations (Expert)
- Human Annotations (Non-Expert)
- Human Annotations (Employees)
- Human Annotations (Contractors)
- Human Annotations (Crowdsourcing)
- Human Annotations (Outsourced / Managed)
- Teams
- Unlabeled
- Others (Please specify)

#### Annotation Characteristic(s)
<!-- scope: periscope -->
<!-- info: Describe relevant characteristics of annotations
as indicated. For quality metrics, consider
including accuracy, consensus accuracy, IRR,
XRR at the appropriate granularity (e.g. across
dataset, by annotator, by annotation, etc.).

Use additional notes to capture any other
relevant information or considerations.

(Usage Note: Duplicate and complete the
following for each annotation type.) -->
**Annotation Type** | **Number**
--- | ---
Number of unique annotations | 123456789
Total number of annotations | 123456789
Average annotations per example | 123456789
Number of annotators per example | 123456789
[Quality metric per granuality] | 123456789
[Quality metric per granuality] | 123456789
[Quality metric per granuality] | 123456789

**Above:** Provide a caption for the above table or visualization.

**Additional Notes:** Add here

#### Annotation Description(s)
<!-- scope: microscope -->
<!-- info: Provide descriptions of the annotations
applied to the dataset. Include links
and indicate platforms, tools or libraries
used wherever possible.

Use additional notes to capture any
other relevant information or
considerations.

(Usage Note: Duplicate and complete
the following for each annotation
type.) -->
**(Annotation Type)**

**Description:** Description of annotations (labels, ratings) produced.
Include how this was created or authored.

**Link:** Relevant URL link.

**Platforms, tools, or libraries:**

- Platform, tool, or library: Write description here
- Platform, tool, or library: Write description here
- Platform, tool, or library: Write description here

**Additional Notes:** Add here

#### Annotation Distribution(s)
<!-- scope: periscope -->
<!-- info: Provide a distribution of annotations for each
annotation or class of annotations using the
format below.

Use additional notes to capture any other
relevant information or considerations.

(Usage Note: Duplicate and complete the
following for each annotation type.) -->
**Annotation Type** | **Number**
--- | ---
Annotations (or Class) | 12345 (20%)
Annotations (or Class) | 12345 (20%)
Annotations (or Class) | 12345 (20%)
Annotations (or Class) | 12345 (20%)
Annotations (or Class) | 12345 (20%)

**Above:** Provide a caption for the above table or visualization.

**Additional Notes:** Add here

#### Annotation Task(s)
<!-- scope: microscope -->
<!-- info: Summarize each task type associated
with annotations in the dataset.

Use additional notes to capture any
other relevant information or
considerations.

(Usage Note: Duplicate and complete
the following for each task type.) -->
**(Task Type)**

**Task description:** Summarize here. Include links if available.

**Task instructions:** Summarize here. Include links if available.

**Methods used:** Summarize here. Include links if available.

**Inter-rater adjudication policy:** Summarize here. Include links if
available.

**Golden questions:** Summarize here. Include links if available.

**Additional notes:** Add here

### Human Annotators
<!-- info: Fill this section if human annotators were used. -->
#### Annotator Description(s)
<!-- scope: periscope -->
<!-- info: Provide a brief description for each annotator
pool performing the human annotation task.

Use additional notes to capture any other
relevant information or considerations.

(Usage Note: Duplicate and complete the
following for each annotation type.) -->
**(Annotation Type)**

**Task type:** Summarize here. Include links if available.

**Number of unique annotators:** Summarize here. Include links if available.

**Expertise of annotators:** Summarize here. Include links if available.

**Description of annotators:** Summarize here. Include links if available.

**Language distribution of annotators:** Summarize here. Include links if
available.

**Geographic distribution of annotators:** Summarize here. Include links if
available.

**Summary of annotation instructions:** Summarize here. Include links if
available.

**Summary of gold questions:** Summarize here. Include links if available.

**Annotation platforms:** Summarize here. Include links if available.

**Additional Notes:** Add here

#### Annotator Task(s)
<!-- scope: microscope -->
<!-- info: Provide a brief description for each
annotator pool performing the human
annotation task.

Use additional notes to capture any
other relevant information or
considerations.

(Usage Note: Duplicate and complete
the following for each annotation
type.) -->
**(Task Type)**

**Task description:** Summarize here. Include links if available.

**Task instructions:** Summarize here. Include links if available.

**Methods used:** Summarize here. Include links if available.

**Inter-rater adjudication policy:** Summarize here. Include links if
available.

**Golden questions:** Summarize here. Include links if available.

**Additional notes:** Add here

#### Language(s)
<!-- scope: telescope -->
<!-- info: Provide annotator distributions for
each annotation type.

Use additional notes to capture any
other relevant information or
considerations.

(Usage Note: Duplicate and
complete the following for each
annotation type.) -->
**(Annotation Type)**

- Language [Percentage %]
- Language [Percentage %]
- Language [Percentage %]

**Above:** Provide a caption for the above table or visualization.

**Additional Notes:** Add here

#### Location(s)
<!-- scope: periscope -->
<!-- info: Provide annotator distributions for each
annotation type.

Use additional notes to capture any other
relevant information or considerations.

(Usage Note: Duplicate and complete the
following for each annotation type.) -->
**(Annotation Type)**

- Location [Percentage %]
- Location [Percentage %]
- Location [Percentage %]

**Above:** Provide a caption for the above table or visualization.

**Additional Notes:** Add here

#### Gender(s)
<!-- scope: microscope -->
<!-- info: Provide annotator distributions for
each annotation type.

Use additional notes to capture any
other relevant information or
considerations.

(Usage Note: Duplicate and complete
the following for each annotation
type.) -->
**(Annotation Type)**

- Gender [Percentage %]
- Gender [Percentage %]
- Gender [Percentage %]

**Above:** Provide a caption for the above table or visualization.

**Additional Notes:** Add here

## Validation Types
<!-- info: Fill this section if the data in the dataset was validated during
or after the creation of your dataset. -->
#### Method(s)
<!-- scope: telescope -->
<!-- info: Select **all applicable**: -->
- Data Type Validation
- Range and Constraint Validation
- Code/cross-reference Validation
- Structured Validation
- Consistency Validation
- Not Validated
- Others (Please Specify)

#### Breakdown(s)
<!-- scope: periscope -->
<!-- info: Provide a description of the fields and data
points that were validated.

Use additional notes to capture any other
relevant information or considerations.

(Usage Note: Duplicate and complete the
following for each validator type.) -->
**(Validation Type)**

**Number of Data Points Validated:** 12345

**Fields Validated**
Field | Count (if available)
--- | ---
Field | 123456
Field | 123456
Field | 123456

**Above:** Provide a caption for the above table or visualization.

#### Description(s)
<!-- scope: microscope -->
<!-- info: Provide a description of the methods used to
validate the dataset.

Use additional notes to capture any other
relevant information or considerations.

(Usage Note: Duplicate and complete the
following for each validator type.) -->
**(Validation Type)**

**Method:** Describe the validation method here. Include links where
necessary.

**Platforms, tools, or libraries:**

- Platform, tool, or library: Write description here
- Platform, tool, or library: Write description here
- Platform, tool, or library: Write description here

**Validation Results:** Provide results, outcomes, and actions taken because
of the validation. Include visualizations where available.

**Additional Notes:** Add here

### Description of Human Validators
<!-- info: Fill this section if the dataset was validated using human
validators -->
#### Characteristic(s)
<!-- scope: periscope -->
<!-- info: Provide characteristics of the validator
pool(s). Use additional notes to capture any
other relevant information or considerations. -->
**(Validation Type)**
- Unique validators: 12345
- Number of examples per validator: 123456
- Average cost/task/validator: $$$
- Training provided: Y/N
- Expertise required: Y/N

#### Description(s)
<!-- scope: microscope -->
<!-- info: Provide a brief description of the validator
pool(s). Use additional notes to capture any
other relevant information or considerations.

(Usage Note: Duplicate and complete the
following for each validator type.) -->
**(Validation Type)**

**Validator description:** Summarize here. Include links if available.

**Training provided:** Summarize here. Include links if available.

**Validator selection criteria:** Summarize here. Include links if available.

**Training provided:** Summarize here. Include links if available.

**Additional Notes:** Add here

#### Language(s)
<!-- scope: telescope -->
<!-- info: Provide validator distributions.
Use additional notes to capture any other relevant information or
considerations.

(Usage Note: Duplicate and complete the following for each annotation type.)-->
**(Validation Type)**

- Language [Percentage %]
- Language [Percentage %]
- Language [Percentage %]

**Above:** Provide a caption for the above table or visualization.

**Additional Notes:** Add here

#### Location(s)
<!-- scope: periscope -->
<!-- info: Provide validator distributions.
Use additional notes to capture any other relevant information or
considerations.

(Usage Note: Duplicate and complete the following for each annotation type.)-->
**(Validation Type)**

- Location [Percentage %]
- Location [Percentage %]
- Location [Percentage %]

**Above:** Provide a caption for the above table or visualization.

**Additional Notes:** Add here

#### Gender(s)
<!-- scope: microscope -->
<!-- info: Provide validator distributions.
Use additional notes to capture any other relevant information or
considerations.

(Usage Note: Duplicate and complete the following for each annotation type.)-->
**(Validation Type)**

- Gender [Percentage %]
- Gender [Percentage %]
- Gender [Percentage %]

**Above:** Provide a caption for the above table or visualization.

**Additional Notes:** Add here

## Sampling Methods
<!-- info: Fill out the following block if your dataset employs any sampling
methods. -->
#### Method(s) Used
<!-- scope: telescope -->
<!-- info: Select **all applicable** methods used in the creation of this
dataset: -->
- Cluster Sampling
- Haphazard Sampling
- Multi-stage Sampling
- Random Sampling
- Retrospective Sampling
- Stratified Sampling
- Systematic Sampling
- Weighted Sampling
- Unknown
- Unsampled
- Others (Please specify)

#### Characteristic(s)
<!-- scope: periscope -->
<!-- info: Provide characteristics of each sampling
method used.

Use additional notes to capture any other
relevant information or considerations.

(Usage Note: Duplicate and complete the
following for each sampling method
used.) -->
**(Sampling Type)** | **Number**
--- | ---
Upstream Source | Write here
Total data sampled | 123m
Sample size | 123
Threshold applied | 123k units at property
Sampling rate | 123
Sample mean | 123
Sample std. dev | 123
Sampling distribution | 123
Sampling variation | 123
Sample statistic | 123

**Above:** Provide a caption for the above table or visualization.

**Additional Notes:** Add here

#### Sampling Criteria
<!-- scope: microscope -->
<!-- info: Describe the criteria used to sample data from
upstream sources.

Use additional notes to capture any other
relevant information or considerations. -->
- **Sampling method:** Summarize here. Include links where applicable.
- **Sampling method:** Summarize here. Include links where applicable.
- **Sampling method:** Summarize here. Include links where applicable.

## Known Applications & Benchmarks
<!-- info: Fill out the following section if your dataset was primarily
created for use in AI or ML system(s) -->
#### ML Application(s)
<!-- scope: telescope -->
<!-- info: Provide a list of key ML tasks
that the dataset has been
used for.

Usage Note: Use comma-separated keywords. -->
*For example: Classification, Regression, Object Detection*

#### Evaluation Result(s)
<!-- scope: periscope -->
<!-- info: Provide the evaluation results from
models that this dataset has been used
in.

Use additional notes to capture any
other relevant information or
considerations.

(Usage Note: Duplicate and complete the
following for each model.) -->
**(Model Name)**

**Model Card:** [Link to full model card]

Evaluation Results

- Accuracy: 123 (params)
- Precision: 123 (params)
- Recall: 123 (params)
- Performance metric: 123 (params)

**Above:** Provide a caption for the above table or visualization.

**Additional Notes:** Add here

#### Evaluation Process(es)
<!-- scope: microscope -->
<!-- info: Provide a description of the evaluation process for
the model's overall performance or the
determination of how the dataset contributes to
the model's performance.

Use additional notes to capture any other relevant
information or considerations.

(Usage Note: Duplicate and complete the following
for each model and method used.) -->
**(Model Name)**

**[Method used]:** Summarize here. Include links where available.

- **Process:** Summarize here. Include links, diagrams, visualizations, tables as relevant.
- **Factors:** Summarize here. Include links, diagrams, visualizations, tables as relevant.
- **Considerations:** Summarize here. Include links, diagrams, visualizations, tables as relevant.
- **Results:** Summarize here. Include links, diagrams, visualizations, tables as relevant.

**Additional Notes:** Add here

#### Description(s) and Statistic(s)
<!-- scope: periscope -->
<!-- info: Provide a description of the model(s) and
task(s) that this dataset has been used
in.

Use additional notes to capture any
other relevant information or
considerations.

(Usage Note: Duplicate and complete the
following for each model.) -->
**(Model Name)**

**Model Card:** Link to full model card

**Model Description:** Summarize here. Include links where applicable.

- Model Size: 123 (params)
- Model Weights: 123 (params)
- Model Layers 123 (params)
- Latency: 123 (params)

**Additional Notes:** Add here

#### Expected Performance and Known Caveats
<!-- scope: microscope -->
<!-- info: Provide a description of the expected performance
and known caveats of the models for this dataset.

Use additional notes to capture any other relevant
information or considerations.

(Usage Note: Duplicate and complete the following
for each model.) -->
**(Model Name)**

**Expected Performance:** Summarize here. Include links where available.

**Known Caveats:** Summarize here. Include links, diagrams, visualizations, and
tables as relevant.

**Additioanl Notes:** Add here

## Terms of Art
### Concepts and Definitions referenced in this Data Card
<!-- info: Use this space to include the expansions and definitions of any
acronyms, concepts, or terms of art used across the Data Card.
Use standard definitions where possible. Include the source of the definition
where indicated. If you are using an interpretation,
adaptation, or modification of the standard definition for the purposes of your
Data Card or dataset, include your interpretation as well. -->
#### Term of Art
Definition: Write here

Source: Write here and share link

Interpretation: Write here

#### Term of Art
Definition: Write here

Source: Write here and share link

Interpretation: Write here

#### Term of Art
Definition: Write here

Source: Write here and share link

Interpretation: Write here

#### Term of Art
Definition: Write here

Source: Write here and share link

Interpretation: Write here

#### Term of Art
Definition: Write here

Source: Write here and share link

Interpretation: Write here

#### Term of Art
Definition: Write here

Source: Write here and share link

Interpretation: Write here

#### Term of Art
Definition: Write here

Source: Write here and share link

Interpretation: Write here

#### Term of Art
Definition: Write here

Source: Write here and share link

Interpretation: Write here

#### Term of Art
Definition: Write here

Source: Write here and share link

Interpretation: Write here

#### Term of Art
Definition: Write here

Source: Write here and share link

Interpretation: Write here

#### Term of Art
Definition: Write here

Source: Write here and share link

Interpretation: Write here

#### Term of Art
Definition: Write here

Source: Write here and share link

Interpretation: Write here

## Reflections on Data
<!-- info: Use this space to include any additional information about the
dataset that has not been captured by the Data Card. For example,
does the dataset contain data that might be offensive, insulting, threatening,
or might otherwise cause anxiety? If so, please contact the appropriate parties
to mitigate any risks. -->
### Title
Write notes here.

### Title
Write notes here.

### Title
Write notes here.
