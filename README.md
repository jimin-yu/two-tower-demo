

파이썬 버전

python 3.9.19
- tensorflow==2.11
- tensorflow-addons==0.20
- tensorflow_recommenders
- mysql-connector-python
- pandas
- scikit-learn
- boto3

### ranking-model의 InferenceService 배포
```
kubectl apply -n kserve-test -f - <<EOF
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: custom-catboost-model
spec:
  predictor:
    containers:
      - name: kserve-container
        image: jjmmyyou111/custom-catboost-model:v2
  transformer:
    containers:
      - image: jjmmyyou111/ranking-transformer:v7
        name: kserve-container
        args:
          - --model_name
          - custom-catboost-model
EOF
```

REQUEST
```
POST http://localhost:8080/v1/models/custom-catboost-model:predict
HOST: custom-catboost-model.kserve-test.example.com
Content-Type: application/json
{
  "instances": [
    {
      "customer_id": "0095c9b47fc950788bb709201f024c5338838a27c59c0299b857f94b504cb9fc",
      "month_sin": 1.2246467991473532e-16,
      "query_emb": [
        0.214135289,
        0.571055949,
        0.330709577,
        -0.225899458,
        -0.308674961,
        -0.0115124583,
        0.0730511621,
        -0.495835781,
        0.625569344,
        -0.0438038409,
        0.263472944,
        -0.58485353,
        -0.307070434,
        0.0414443575,
        -0.321789205,
        0.966559
      ],
      "month_cos": -1
    }
  ]
}
```


RESPONSE
```
{
  "ranking": [
    [
      0.6303086438567949,
      "596097001"
    ],
    [
      0.6260879998025899,
      "608007006"
    ],
    [
      0.602544614079735,
      "675827003"
    ],
    [
      0.601574127398539,
      "625311009"
    ],
    [
      0.601529882743909,
      "658321004"
    ],
    [
      0.5937195210744586,
      "436570008"
    ],
    [
      0.5838643667616848,
      "615021009"
    ],
    [
      0.5677551770392755,
      "493438019"
    ],
    [
      0.5334478929879939,
      "564786020"
    ],
    [
      0.5313247993825141,
      "710059001"
    ],
    [
      0.5123185170602064,
      "699461001"
    ],
    [
      0.504259190115954,
      "564786023"
    ],
    [
      0.4958446632768758,
      "625316004"
    ],
    [
      0.46320849437326467,
      "637566003"
    ],
    [
      0.42730690486438494,
      "651273002"
    ],
    [
      0.4249618500178985,
      "637820002"
    ],
    [
      0.41242867714305065,
      "669105001"
    ],
    [
      0.3969789858405491,
      "704048003"
    ],
    [
      0.3791290627297427,
      "606712001"
    ],
    [
      0.3791290627297427,
      "581228001"
    ],
    [
      0.2598100906865275,
      "663383001"
    ],
    [
      0.20798648640676326,
      "664228002"
    ]
  ]
}
```
### query InferenceService 배포
secret
```
kubectl apply -n kserve-test -f - <<EOF
apiVersion: v1
kind: Secret
metadata:
  name: aws-s3-secret
type: Opaque
data:
  AWS_ACCESS_KEY_ID: {{base64 encoded}}
  AWS_SECRET_ACCESS_KEY: {{base64 encoded}}
EOF
```
service account
```
kubectl apply -n kserve-test -f - <<EOF
apiVersion: v1
kind: ServiceAccount
metadata:
  name: sa
secrets:
- name: aws-s3-secret
EOF
```
inference service
```
kubectl apply -n kserve-test -f - <<EOF
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: query
spec:
  predictor:
    serviceAccountName: sa
    model:
      modelFormat:
        name: tensorflow
      storageUri: "s3://jimin-model/query/1.tar.gz"
  transformer:
    containers:
      - image: jjmmyyou111/query-transformer:v5
        name: kserve-container
        args:
          - --model_name
          - query
EOF
```

REQUEST
```
{
  "instances": {
    "customer_id": "048962db9aca38ca4b98c70880a44b60f12562c8d4df5e34457401c14ec0dcbe",
    "month_of_purchase": "2022-11-15T12:16:25.330916"
  }
}
```

RESPONSE
```
{
  "predictions": {
    "ranking": [
      [
        0.5519261871389344,
        "501616009"
      ],
      [
        0.5240805173245946,
        "501616048"
      ],
      [
        0.5012625639167201,
        "650539001"
      ],
      [
        0.5012625639167201,
        "501619006"
      ],
      [
        0.5012625639167201,
        "501616007"
      ],
      [
        0.5012625639167201,
        "443078001"
      ],
      [
        0.49816390167441205,
        "624119001"
      ],
      [
        0.48211449081483854,
        "501616028"
      ],
      [
        0.46911498995736184,
        "649299001"
      ],
      [
        0.42329270637100214,
        "634037002"
      ],
      [
        0.4060623996653744,
        "443078003"
      ],
      [
        0.40306121772910936,
        "518980005"
      ],
      [
        0.36609407649364495,
        "467302079"
      ],
      [
        0.34719488266949755,
        "492171003"
      ],
      [
        0.3471665599639389,
        "638751005"
      ]
    ]
  }
}
```