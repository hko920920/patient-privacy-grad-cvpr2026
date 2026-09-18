from pathlib import Path
import json, yaml
root=Path(__file__).resolve().parent
contract={
 'schema':'prrd-study-contract/v3-draft',
 'status':'PROPOSED_VALUES_NOT_RUNTIME_FROZEN',
 'created':'2026-09-18',
 'repo':{'name':'hko920920/patient-privacy-grad-cvpr2026','commit':'e560b2c2b6a8413c842d5e691a20b19be2145237'},
 'execution':{'new_patient_model_runs_performed':0,'gpu_authorized':False,'dp_authorized':False,
              'expert_authorized':False,'reserved_authorized':False,'runtime_implemented':False},
 'data':{
  'P':{'patients':672,'images':813,'positive_patients':6,'mixed_patients':3,'role':'public','manifest':None,'manifest_sha256':None},
  'Q':{'patients':2027,'images':5097,'positive_patients':114,'mixed_patients':102,'role':'protected_former_selection_already_used_for_calibration_and_Rwide','manifest':None,'manifest_sha256':None},
  'V':{'patients':2026,'images':5047,'role':'reused_development_not_independent_confirmation','manifest':None,'manifest_sha256':None},
  'expert':{'patients':532,'images':810,'access':'LOCKED'},
  'reserved':{'patients':4213,'access':'LOCKED'},
  'Q80':{'patients':80,'images':320,'access':'NOT_USED_IN_THIS_BRANCH'}},
 'source_encoder':{'family':'BioViL-T official IMAGE model','model_checkpoint_sha256':None,
                   'source_code_commit':None,'embedding_layer':None,'preprocessing_sha256':None,
                   'feature_projection':'P-only PCA, no whitening; saved deterministic basis',
                   'projection_dimension':16,'projection_sha256':None,'norm_cap':1.0,
                   'norm_scale':'P-only q95 projected norm with positive floor',
                   'eval_mode':True,'trainable_parameters':False,'synthetic_input_gradients':True},
 'recipient_primary':{'family':'torchvision DenseNet121','weights_enum':'IMAGENET1K_V1',
                      'checkpoint_sha256':None,'embedding':'pre-classifier global pooled features',
                      'projection':'separate P-only PCA128 + norm cap','projection_sha256':None,
                      'role':'not used for synthesis gradients/checkpoint/parameter selection',
                      'preprocess':'official pinned weight transform; not silently replaced'},
 'recipient_final_reserved':{'family':'torchvision ViT-B/16','weights_enum':'IMAGENET1K_V1',
    'checkpoint_sha256':None,'embedding':'pre-head CLS feature','projection':'P-only PCA128 + norm cap',
    'development_performance_opened':False,'role':'confirmation recipient; frozen before final and not a tuning target'},
 'recipient_secondary':{'family':'existing ImageNet ResNet18 full BCE finetuning',
                        'role':'image-only compatibility, not mandatory relation-aware transfer success',
                        'classifier_seeds':[11,23,37],'updates_per_run':400,
                        'real_examples_per_batch':16,'synthetic_examples_per_batch':16},
 'risk':{'name':'class-balanced patient-average squared point + mixed-patient squared contrast + ridge',
         'beta_primary':1.0,'beta_ablation':0.0,'ridge':0.1,'intercept':False,
         'patient_delta':'(positive centroid - negative centroid)/2',
         'primary_target_distribution':'class-specific patient average over P union Q'},
 'arms':{
  'A_PUBLIC_POINT':{'mode':'point','data':'P','marginal_images':128,'relation_pairs':0},
  'B_PRIVATE_POINT':{'mode':'point','data':'P+Q','marginal_images':128,'relation_pairs':0},
  'C_PRRD':{'mode':'relation','data':'P+Q','marginal_images':64,'relation_pairs':32},
  'D_Q_SHUFFLE':{'mode':'relation','data':'P+Q with Q-only mixed centroid permutation','marginal_images':64,'relation_pairs':32,'dp_allowed':False},
  'R_JOINT_PAIR':{'mode':'joint_pair','data':'P+Q','marginal_images':64,'relation_pairs':32,'provenance':'standard joint-moment adapted comparator, NOT original CovMatch reproduction'}},
 'synthesis':{'bank_seeds':[101,202,303],'total_images_per_bank':128,'successful_updates':500,
              'optimizer':'AdamW','lr':0.01,'weight_decay':0.0,'betas':[0.9,0.999],
              'precision':'FP32','renderer':'bounded grayscale logit residual pyramid',
              'resolutions':[8,16,32,64,112,224],'activation_steps':[1,81,161,241,321,401],
              'initial_residual_std':0.001,'template_source':'P only, outcome-independent patient hash',
              'loss':{'point':1.0,'relation':1.0,'pixel_anchor':0.01,'total_variation':0.0001,'augmentation_regularizer':0.1},
              'augmentation':{'rotation_degrees':3,'translation_fraction':0.01,'scale':[0.98,1.02],
                              'brightness':[0.98,1.02],'flip':False,'shared_within_pair':True,
                              'interpretation':'synthetic regularizer; not an assertion of real augmented moment equality'},
              'export':{'format':'8bit grayscale PNG','resolution':224,'evaluate_exported_pixels_only':True},
              'microbatch':None,'final_checkpoint_only':True,'source_recipient_outcome_selection':False},
 'runtime_freeze_required':[
  'all P/Q/V manifests and SHA256', 'source image weights/layer/preprocessing and code SHA',
  'recipient weights and public-only projection SHA','public templates and shuffle schedule',
  'public-input parity + whole-bank throughput/peak-memory','bounded query and count normalization',
  'actual hardware and run-time budget approval','all baseline implementation scopes'],
 'non_dp_package':{'arms':5,'bank_repeats':3,'banks':15,'images':1920,'synthesis_updates':7500,
                   'deterministic_synthetic_readouts':48,'trusted_real_readouts':6,
                   'resnet18_runs':45,'resnet18_updates':18000,'authorized':False},
 'statistics':{'primary_metric':'image-level AUROC','secondary_metric':'average precision',
               'cluster_unit':'patient','paired_draws_across_arms':True,'development_draws':2000,
               'mean_metrics_not_prediction_ensemble':True,
               'variation_levels':['test patients','synthetic bank initialization','classifier seed','DP mechanism noise'],
               'development_intervals':'conditional descriptive; not fresh confirmation',
               'non_dp_engineering_rules':{'recipient_C_minus_B_mean_AUROC':0.01,
                  'recipient_C_minus_D_mean_AUROC_strictly_above':0.0,
                  'positive_bank_differences_minimum':2,'bank_count':3,
                  'C_AP_not_lower_than_B':True,'report_A_and_public_real_and_R_joint':True},
               'final':{'draws':10000,'planned_primary_family_size':3,
                  'per_comparison_two_sided_confidence':1-0.05/3,'locked':True}},
 'privacy':{'unit':'all records of one patient','adjacency':'add_remove',
            'eps_primary':8.0,'eps_secondary':4.0,'delta':0.00001,
            'add_remove_sensitivity':{'point':6**0.5,'relation':3.0,'joint_pair':3.0},
            'dimension_d16':{'point':306,'relation':459,'joint_pair':867},
            'noisy_count_denominator_floor':1.0,
            'synthesis_moment_projection_default':False,
            'direct_predictor_repair':'symmetrize H; floor eigenvalues at 0; add ridge; solve',
            'production_noise_sampler':'NOT_IMPLEMENTED; secure deployment module required',
            'production_noise_seed_public':False,'independent_noise_across_distinct_releases':True,'composition_required_for_multiple_releases':True,
            'same_release_multiple_banks':'postprocessing only when no additional private access',
            'no_retrospective_protection_of_nonDP_outputs':True,
            'no_nonDP_private_HPO_as_free_public_choice':True,
            'mechanism_replicates':3,'initializations_per_release':3,
            'DP1_epsilon8_banks':27,'DP2_epsilon4_banks':27,
            'DP2_automatic':False},
 'claim_limits':['not clinically validated synthetic labels','not causal disease changes',
                 'not every visit-pair/longitudinal trajectory preservation',
                 'not first DP/covariance/pairwise synthesis in general',
                 'not empirical superiority established by algebra tests']
}
(root/'experiment_contract_template.yaml').write_text(
 '# Proposed research execution values. Nulls must be bound locally before freezing.\n'
 '# This template does not authorize model/DP/final execution.\n'+
 yaml.safe_dump(contract,sort_keys=False,allow_unicode=True),encoding='utf-8')
runs=[]
for arm in contract['arms']:
 for seed in contract['synthesis']['bank_seeds']:
  runs.append({'phase':'W2_NONDP','arm':arm,'bank_seed':seed,'mechanism_rep':None,'epsilon':None,
               'images':128,'updates':500,'status':'PLANNED_NOT_EXECUTED'})
for eps,phase in [(8,'W4_DP1'),(4,'W4_DP2_CONDITIONAL')]:
 for arm in ['B_PRIVATE_POINT','C_PRRD','R_JOINT_PAIR']:
  for noise_rep in [1,2,3]:
   release_id=f'{phase}_{arm}_mechanism_{noise_rep}'
   for seed in [101,202,303]:
    runs.append({'phase':phase,'arm':arm,'bank_seed':seed,'mechanism_rep':noise_rep,'epsilon':eps,
      'release_id':release_id,'public_noise_seed':None,'images':128,'updates':500,
      'status':'PLANNED_NOT_EXECUTED_NOT_AUTHORIZED'})
(root/'planned_run_matrix.json').write_text(json.dumps({'scope':'bank plan only; no execution',
    'run_count':len(runs),'runs':runs},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
assert len(runs)==15+27+27
assert sum(r['updates'] for r in runs if r['phase']=='W2_NONDP')==7500
print('Wrote contract +',len(runs),'planned banks; 0 executed.')
