# mpirun -np 6 python -u train_HER.py --env-name='FetchSlide' --n-epochs=100
# python collect_distance_data.py --episodes=50000 --env-name=FetchPickAndPlace --agent-location=saved_models/her_FetchPickAndPlace.pkl --max-steps=35
# python collect_distance_data.py --episodes=50000 --env-name=FetchSlide --agent-location=saved_models/her_FetchSlide.pkl --max-steps=35
# python train_distance_function.py  --batch-size=1024  --cuda --env-name=FetchPickAndPlace --hidden-size=256 --epoch=100 --dropout-rate=.5
# python train_distance_function.py  --batch-size=1024  --cuda --env-name=FetchSlide --hidden-size=256 --epoch=100 --dropout-rate=.5



# mpirun -np 3 python -u train_HER.py --env-name='FetchReach' --n-epochs=10 
mpirun -np 3 python -u train_HER.py --env-name='FetchReach' --n-epochs=10 --p2p
python main.py FetchReach stable-sparse-rrt gdsst psst pgdsst
# mpirun -np 3 python -u train_HER.py --env-name='FetchPush' --n-epochs=100 
mpirun -np 3 python -u train_HER.py --env-name='FetchPush' --n-epochs=100 --p2p
python main.py FetchPush stable-sparse-rrt gdsst psst pgdsst
# mpirun -np 3 python -u train_HER.py --env-name='FetchPickAndPlace' --n-epochs=200
mpirun -np 3 python -u train_HER.py --env-name='FetchPickAndPlace' --n-epochs=200 --p2p
python main.py FetchPickAndPlace stable-sparse-rrt gdsst psst pgdsst

# python collect_distance_data_p2p.py --env-name=FetchPickAndPlace --episodes=10000 --agent-location=saved_models/her_FetchPickAndPlace_p2p.pkl
# python train_distance_function.py --env-name=FetchPickAndPlace --epochs=200 --p2p --agent-location=saved_models/her_FetchPickAndPlace.pkl
# python collect_distance_data_p2p.py --env-name=FetchPush --episodes=10000 --agent-location=saved_models/her_FetchPush_p2p.pkl
# python train_distance_function.py --env-name=FetchPush --epochs=200 --p2p --agent-location=saved_models/her_FetchPush.pkl
# python collect_distance_data_p2p.py --env-name=FetchReach --episodes=10000 --agent-location=saved_models/her_FetchReach_p2p.pkl
# python train_distance_function.py --env-name=FetchReach --epochs=200 --p2p --agent-location=saved_models/her_FetchReach.pkl
# python collect_distance_data_p2p.py --env-name=FetchSlide --episodes=10000 --agent-location=saved_models/her_FetchReach_p2p.pkl
# python train_distance_function.py --env-name=FetchSlide --epochs=200 --p2p --agent-location=saved_models/her_FetchReach.pkl