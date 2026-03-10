<h1 align="center">
  SMS / Email Spam Detection System
</h1>

<h1 align="center">

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.1-orange?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation PDF](https://img.shields.io/badge/📘%20Project_Documentation-PDF-blue)](https://github.com/apdoolhamza/SMS-Email-Spam-Detector/blob/main/Docs/SMS_Email_Spam_Classifier.pdf)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/apdoolhamza/sms-email-spam-classifier)
[![Live Demo on Hugging Face](https://img.shields.io/badge/🤗%20Live%20Demo-Hugging%20Face-yellow)](https://huggingface.co/spaces/apdoolhamza/SMS-Email-Spam-Detector)

</h1>

A lightweight, production-ready Spam Classifier for SMS and short emails. Achieves F1-score ~0.88 on spam class with extremely low false positives using scikit-learn.

Built with clean engineering practices perfect for real-world deployment in telecom, email gateways, or personal security tools.

<p align="center">
  <a href="https://huggingface.co/spaces/apdoolhamza/SMS-Email-Spam-Detector">
    <img src="Images/screenshot.jpg" width="700"/>
  </a>
</p>

## Key Features

- Professional text cleaning + domain-specific engineered features  
- TF-IDF + numeric feature pipeline (length, keyword flags, punctuation)  
- HalvingGridSearchCV hyperparameter tuning  
- Isotonic probability calibration for trustworthy confidence scores  
- Full evaluation suite (ROC, PR curves, confusion matrix)  
- Model saved with joblib (fast loading & compression)  
- Ready for deployment  

## Demo

[![Live Demo on Hugging Face](https://img.shields.io/badge/🤗%20Live%20Demo-Hugging%20Face-yellow)](https://huggingface.co/spaces/apdoolhamza/SMS-Email-Spam-Detector)

## Results Summary

| Metric              | Value (Spam class) |
|---------------------|--------------------|
| F1-score            | ~ 0.88             |
| Precision           | ~ 0.92             |
| Recall              | ~ 0.8591           |
| ROC-AUC             | > 0.97             |
| Inference speed     | < 10 ms / message  |

## Installation & Usage

```bash
pip install -r requirements.txt
```

## Why This Project Matters

In 2026, spam and AI-powered phishing cost businesses billions annually. This system demonstrates how a well-engineered classic ML solution can deliver enterprise-grade performance with minimal resources.

## License
This project is licensed under the MIT License – see the LICENSE file for details.

## Acknowledgments

* UCI SMS Spam Collection Dataset
* scikit-learn community

## Contact / Contributing

Feel free to open an issue or submit a pull request.

## Author

```
Apdoolmajeed Hamza (apdoolhamza)
AI/ML Engineer | Full-stack Web Developer
```
- LinkedIn: https://www.linkedin.com/in/apdoolhamza/
- GitHub:   https://github.com/apdoolhamza/
