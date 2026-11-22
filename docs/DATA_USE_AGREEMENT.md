# License-plate video dataset: data-use agreement

The license-plate (LP) video dataset used in *Investigating the Resampling-Kernel Gap in Video Super-Resolution for
License Plate Recognition* (Sathiyendra et al., submitted to IEEE T-ITS) is not distributed openly: its frames show
readable license plates, and copyright in the source videos remains with their owners. Researchers can obtain it for
non-commercial research by accepting this agreement.

## How to request access

E-mail the corresponding author, Dr. T. D. Ambegoda (thanuja@cse.mrt.ac.lk), from an institutional address with:

1. this agreement, completed and signed (section 3);
2. a short description of the intended research (two or three sentences);
3. for students: the name and signature of a supervisor at the same institution, who becomes responsible for the
   student's compliance.

We reply with a download link. Each research group signs once; everyone in the group who handles the data must be
named in section 3.

## 1. The data

* HR frames (256 × 256 crops centered on the plate) and LR frames (64 × 64, OpenCV cubic ×4 downsampling) of 282 clips:
  221 training, 30 validation, 31 test;
* the splits (`splits/lp_lpsr.json` in the code repository);
* hand-verified plate strings for the 51 evaluation clips.

The clips are crops of publicly available traffic and CCTV videos on YouTube, made by the authors.

## 2. Terms

By signing, the recipient agrees to the following.

1. **Purpose.** The data are used only for non-commercial research and teaching on image and video enhancement and
   license plate recognition. They are not used to develop or improve a commercial product or service.
2. **No re-identification.** The recipient does not try to identify, locate or track any vehicle, vehicle keeper or
   person shown in the data, and does not link the data with other data for that purpose. The data are not used for
   surveillance, law enforcement or any decision about an individual.
3. **No redistribution.** The recipient does not share, publish or host the data, or any subset or modified version
   from which frames or plate strings can be recovered, outside the people named in section 3. Others must sign their
   own agreement. Models trained on the data, code, and aggregate results may be shared.
4. **Publications.** Papers may show a small number of example frames or crops where this is needed to explain the
   results. They cite the paper above.
5. **Security.** The data are stored on access-controlled systems and are not placed in public repositories, public
   cloud buckets or shared drives open to others.
6. **Removal requests.** If the authors ask for a clip to be removed, for example at the request of a rights holder or
   a person shown, the recipient deletes it within 30 days.
7. **Deletion.** The recipient deletes the data when the research ends, when this agreement is breached, or when the
   authors ask.
8. **No warranty.** The data are provided as they are, without any warranty. The authors grant no rights in the source
   videos. The recipient is responsible for complying with the law that applies to them, including data-protection
   law.

## 3. Signature

| | Recipient | Supervisor (students only) |
|---|---|---|
| Name | | |
| Position | | |
| Institution | | |
| Institutional e-mail | | |
| Other people who will handle the data | | |
| Intended research (2–3 sentences) | | |
| Date | | |
| Signature | | |
