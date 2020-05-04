import json
import os
import random

random.seed(2020)


def make_target_dir(target_dir):
  if not os.path.exists(target_dir):
    os.makedirs(target_dir)
  return target_dir


def preprocess(target_dir, augment=0):

  def _convert(indict, answerKey_map, target_mode='full_text', choice_first=False, shuffle=False):
    outdict = {'id': indict['id']}
    n_choices = len(indict['question']['choices'])

    question_type = indict['id'].split('_')[0]
    question_body = indict['question']['stem']
    choices_body = [indict['question']['choices'][i]['text']
                    for i in range(n_choices)]
    answerKey = answerKey_map[indict['answerKey']]
    if shuffle:
      choice_keys = [i for i in range(n_choices)]
      random.shuffle(choice_keys)
      answerKey = str(choice_keys.index(int(answerKey) - 1) + 1)
      choices_body = [choices_body[i] for i in choice_keys]

    full_choice_body = [''.join(
        ['choice', str(i + 1), ': ', choices_body[i]]) for i in range(n_choices)]

    if choice_first:
      outdict['input_text'] = ' '.join(
          [question_type] + full_choice_body + ['question:', question_body])
    else:
      outdict['input_text'] = ' '.join(
          [question_type, 'question:', question_body] + full_choice_body)

    if indict['answerKey'] in answerKey_map:

      if target_dir != '../data4T5':
        outdict['target_text'] = int(answerKey) - 1
      elif target_mode == 'full_text':
        outdict['target_text'] = ''.join(
            ['choice', answerKey, ': ', choices_body[int(answerKey) - 1]])
      elif target_mode == 'choice_only':
        outdict['target_text'] = 'choice' + answerKey
      else:  # index output
        raise ValueError(f"No such target_mode choice {target_mode}")
    else:
      raise ValueError(f"No such choice {indict['answerKey']}")

    return outdict

  source_dir = '../ARC-OBQA-RegLivEnv-IR10V2'
  answerKey_map = {'A': '1', 'B': '2', 'C': '3', 'D': '4',
                   'E': '5', '1': '1', '2': '2', '3': '3', '4': '4', '5': '5'}
  for file in os.listdir(source_dir):
    with open(os.path.join(source_dir, file), 'r') as infile:
      with open(os.path.join(target_dir, file), 'w') as outfile:
        for line in infile:
          problem = _convert(json.loads(line), answerKey_map)
          json.dump(problem, outfile)
          outfile.write('\n')

          if file == 'train.jsonl' or file == 'dev.jsonl':
            for _ in range(augment):
              problem = _convert(json.loads(line), answerKey_map, shuffle=True)
              json.dump(problem, outfile)
              outfile.write('\n')


def split_dev(target_dir):
  '''split dev set into ARC Challenge and ARC Easy'''
  with open(os.path.join(target_dir, 'test.jsonl'), 'r') as in_file:
    with open(os.path.join(target_dir, 'arcch_test.jsonl'), 'w') as ch_file:
      with open(os.path.join(target_dir, 'arcez_test.jsonl'), 'w') as ez_file:

        for line in in_file:
          d = json.loads(line)
          if d['id'][:5] == 'ARCCH':
            json.dump(d, ch_file)
            ch_file.write('\n')
          elif d['id'][:5] == 'ARCEZ':
            json.dump(d, ez_file)
            ez_file.write('\n')


# def preprocess_race(target_dir):
#   def _convert(indict):
#     q_str = []
#     for i, q in enumerate(indict['questions']):


if __name__ == '__main__':
  # target_dir = make_target_dir('../data4T5')
  target_dir = make_target_dir('../data4lsh')
  make_target_dir(target_dir)
  preprocess(target_dir, augment=0)
  split_dev(target_dir)
  # preprocess_race(target_dir)
