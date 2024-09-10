# Data

## Instruction Data

Our training data are consist of video instruction and image instruction. You can download our preprocessed instruction data from [here](https://huggingface.co/datasets/ColorfulAI/VideoLLaMB-IT)

1. video instruction

We use the video instruction data from [PLLaVA](https://github.com/magic-research/PLLaVA)

**Instruction**: You can download from [magic_jsons](https://huggingface.co/datasets/cathyxl/magic_jsons)

**Videos**: 

*Note: The Preprocessed links come from this [issue](https://github.com/OpenGVLab/Ask-Anything/issues/176) provided by the authors of VideoChat2. If the links are not available, please refer to their original [VideoChat2](https://github.com/OpenGVLab/Ask-Anything/tree/main/video_chat2).*

- [VideoChat](https://github.com/OpenGVLab/InternVideo/tree/main/Data/instruction_data) Preprocessed [download link](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/videochat2/data/videochat2_conversation_videos.zip)
- [VideoChatGPT](https://github.com/mbzuai-oryx/Video-ChatGPT/tree/main/data) Direct [download link](https://mbzuaiac-my.sharepoint.com/personal/hanoona_bangalath_mbzuai_ac_ae/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fhanoona%5Fbangalath%5Fmbzuai%5Fac%5Fae%2FDocuments%2FVideo%2DChatGPT%2FData%5FCode%5FModel%5FRelease%2FData&ga=1)
- [Kinetics-710](https://github.com/OpenGVLab/UniFormerV2/blob/main/DATASET.md) Alternative [download link](https://openxlab.org.cn/datasets?keywords=kinetics)
- [SthSthV2](https://developer.qualcomm.com/software/ai-datasets/something-something)
- [NExTQA](https://github.com/doc-doc/NExT-QA)
- [CLEVRER](https://clevrer.csail.mit.edu/) Direct [download links](https://github.com/OpenGVLab/Ask-Anything/issues/176#issuecomment-2121805009)
- [YouCook2](https://youcook2.eecs.umich.edu/) Preprocessed [download link](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/videochat2/data/youcook_split_videos.zip)
- [TextVR](https://github.com/callsys/textvr)
- [TGIF](https://github.com/YunseokJANG/tgif-qa)
- [EgoQA](https://ego4d-data.org/) Preprocessed [download link](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/videochat2/data/egoqa_split_videos.zip)


<!-- For the complete VideoChat2 Videos, you can download the videos of WebVid dataset from: 

- [WebVid](https://maxbain.com/webvid-dataset/) -->

2. image instruction

**Instruction**: [LLaVA-Instruct](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K)

**Images**: [LLaVA-Instruct](https://github.com/haotian-liu/LLaVA?tab=readme-ov-file#visual-instruction-tuning)


We mix the video instruction and image instruction. These files are organized in the following structure

```bash
playground/data/
    ├── llava_videochat2_filter.json
    ├── video/videochat
        ├── clevrer/video_train
        ├── egoqa
        ├── kinetic
            ├── k400
            ├── k600
            └── k700
        ├── nextqa
        ├── ssv2
        ├── textvr
            ├── Activity
            ├── Cooking
            ├── Driving
            ├── Games
            ├── News_Movie
            ├── Sports
            ├── Street_View_Indoor
            └── Street_View_Outdoor
        ├── tgif
        ├── videochat2
        ├── videochatgpt
        └── youcook
            ├── train
            └── validation
    ├── image
        ├── coco/train2017
        ├── gqa/images
        ├── ocr_vqa/images
        ├── textvqa/train_images
        └── vg
            ├── VG_100K
            └── VG_100K_2
```


## Evaluation Data

- [EgoScheme-subset](https://egoschema.github.io/)

- [NExTQA-val](https://egoschema.github.io/)

- [EgoPlan-test](https://github.com/ChenYi99/EgoPlan?tab=readme-ov-file#egoplan-evaluation-data): We cut the video clips by the `start_frame` and `end_frame`.  Preprocessed [download link](https://huggingface.co/datasets/ColorfulAI/EgoPlan_test)

- [MVBench](https://huggingface.co/datasets/OpenGVLab/MVBench)

 
These files are organized in the following structure

```bash
playground/eval/GPT_Zero_Shot_QA
    ├── EgoSchema_Zero_Shot_QA
        ├── videos
        ├── test_q.json
        └── test_a.json
    ├── NExT_Zero_Shot_QA
        ├── videos
        ├── test_q.json
        └── test_a.json
    ├── EgoPlan_Zero_Shot_QA
        ├── videos
        ├── test_q.json
        └── test_a.json
    └── MVBench_Zero_Shot_QA
        ├── videos
            ├── clevrer
            ├── FunQA_test
            ├── Moments_in_Time_Raw
            ├── nturgbd
            ├── perception
            ├── scene_qa
            ├── ssv2_video
            ├── sta
            ├── star
            ├── tvqa
            └── vlnqa
        ├── test_q.json
        └── test_a.json
```




To fit our evaluation pipeline, we reformat these texts into two files, `test_q.json` and `test_a.json`, in the following format:

test_q.json

```json
[
    {
        "video_name": "",
        "question_id": "",
        "question": "",
        "option":{
            "option 0": "",
            "option 1": "",
            "option 2": "",
            "option 3": "",
            "option 4": "",
        },
        "type": ""
    }
]
```

test_a.json

```json
[
    {
        "answer": 0,
        "question_id": "",
    }
]
```
