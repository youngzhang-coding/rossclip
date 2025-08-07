# Modified from https://github.com/360CVGroup/FG-CLIP/blob/main/fgclip/eval/coco_retrieval.py

import torch

def eval_coco(model, coco, image_processor, tokenizer, device, args):
    image_features = []

    text_features = []

    pred_true = 0

    image_size = args.image_size

    with torch.no_grad():
        index = 0
        for image, captions in coco:
            image = image.resize((image_size, image_size))
            image_input = image_processor.preprocess(image, return_tensors="pt")[
                "pixel_values"
            ].to(device)
            image_feature = model.get_image_features(image_input)
            image_features.append(image_feature)

            captions = captions[0:5]

            caption_input = torch.tensor(
                tokenizer(
                    captions,
                    max_length=args.max_length,
                    padding="max_length",
                    truncation=True,
                ).input_ids,
                dtype=torch.long,
                device=device,
            )

            if args.max_length > 100:
                walk_short_pos = False

            text_feature = model.get_text_features(caption_input)

            text_features.extend(text_feature)

            index += 1

            print(index, ": ", len(coco))

            image_features = torch.stack(image_features).squeeze()
            image_features /= image_features.norm(dim=-1, keepdim=True)

            text_features = torch.stack(text_features)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            similarity = image_features.squeeze() @ text_features.squeeze().T

            print("I2T")

            for i in range(5000):
                pred = similarity[i]
                b = pred.argsort()[-1:]
                for j in range(5):
                    true_index = 5 * i + j
                    if true_index in b:
                        pred_true = pred_true + 1
                        break
            print("(Image2Text)With top 1 as correct answer: {}", pred_true / 5000)
            pred_true = 0

            for i in range(5000):
                pred = similarity[i]
                b = pred.argsort()[-5:]
                for j in range(5):
                    true_index = 5 * i + j
                    if true_index in b:
                        pred_true = pred_true + 1
                        break
            print("(Image2Text)With top 5 as correct answer: {}", pred_true / 5000)
            pred_true = 0

            for i in range(5000):
                pred = similarity[i]
                b = pred.argsort()[-10:]
                for j in range(5):
                    true_index = 5 * i + j
                    if true_index in b:
                        pred_true = pred_true + 1
                        break
            print("(Image2Text)With top 10 as correct answer: {}", pred_true / 5000)
            pred_true = 0

            print("T2I")
            similarity = similarity.T
            for i in range(25000):
                pred = similarity[i]
                b = pred.argsort()[-1:]
                true_index = i // 5
                if true_index in b:
                    pred_true = pred_true + 1

            print("(Text2Image)With top 1 as correct answer: {}", pred_true / 25000)
            pred_true = 0

            for i in range(25000):
                pred = similarity[i]
                b = pred.argsort()[-5:]
                true_index = i // 5
                if true_index in b:
                    pred_true = pred_true + 1

            print("(Text2Image)With top 5 as correct answer: {}", pred_true / 25000)
            pred_true = 0

            for i in range(25000):
                pred = similarity[i]
                b = pred.argsort()[-10:]
                true_index = i // 5
                if true_index in b:
                    pred_true = pred_true + 1

            print("(Text2Image)With top 10 as correct answer: {}", pred_true / 25000)